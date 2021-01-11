import tensorflow as tf

import model

def top_k_logits(logits, k):
    if k == 0:
        # no truncation
        return logits

    def _top_k():
        values, _ = tf.nn.top_k(logits, k=k)
        min_values = values[:, -1, tf.newaxis]
        return tf.compat.v1.where(
            logits < min_values,
            tf.ones_like(logits, dtype=logits.dtype) * -1e10,
            logits,
        )
    return tf.cond(
       pred=tf.equal(k, 0),
       true_fn=lambda: logits,
       false_fn=lambda: _top_k(),
    )


def top_p_logits(logits, p):
    with tf.compat.v1.variable_scope('top_p_logits'):
        logits_sort = tf.sort(logits, direction='DESCENDING')
        probs_sort = tf.nn.softmax(logits_sort)
        probs_sums = tf.cumsum(probs_sort, axis=1, exclusive=True)
        logits_masked = tf.compat.v1.where(probs_sums < p, logits_sort, tf.ones_like(logits_sort)*1000) # [batchsize, vocab]
        min_logits = tf.reduce_min(input_tensor=logits_masked, axis=1, keepdims=True) # [batchsize, 1]
        return tf.compat.v1.where(
            logits < min_logits,
            tf.ones_like(logits, dtype=logits.dtype) * -1e10,
            logits,
        )


def sample_sequence(*, hparams, length, start_token=None, batch_size=None, context=None, temperature=1, top_k=0, top_p=0.0):
    if start_token is None:
        assert context is not None, 'Specify exactly one of start_token and context!'
    else:
        assert context is None, 'Specify exactly one of start_token and context!'
        context = tf.fill([batch_size, 1], start_token)

    def step(hparams, tokens, past=None):
        lm_output = model.model(hparams=hparams, X=tokens, past=past, reuse=tf.compat.v1.AUTO_REUSE)

        print("TEST STEP")
        logits = lm_output['logits'][:, :, :hparams.n_vocab]
        presents = lm_output['present']
        presents.set_shape(model.past_shape(hparams=hparams, batch_size=batch_size))
        print("TS 2")
        return {
            'logits': logits,
            'presents': presents,
            'attentions': lm_output['attentions']
        }

    with tf.compat.v1.name_scope('sample_sequence'):
        # Don't feed the last context token -- leave that to the loop below
        # TODO: Would be slightly faster if we called step on the entire context,
        # rather than leaving the last token transformer calculation to the while loop.
        context_output = step(hparams, context[:, :-1])
        print("Context Output (single) is done")

        
        # attentions_stack_array = tf.zeros([0, batch_size, hparams.n_layer, hparams.n_head, 1, hparams.n_ctx])
        attentions_stack_array = tf.zeros([batch_size, 0, hparams.n_layer, 1, hparams.n_ctx])


        def body(past, prev, output, attn_stack_array):
            next_outputs = step(hparams, prev[:, tf.newaxis], past=past)
            logits = next_outputs['logits'][:, -1, :]  / tf.cast(temperature, dtype=tf.float32)

            attentions = next_outputs['attentions']

            attentions = tf.reduce_max(attentions, axis=2)
            # paddings = [[0,0], [0,0], [0,0], [0,0], [0, hparams.n_ctx - tf.shape(attentions)[-1]]]
            paddings = [[0,0], [0,0], [0,0], [0, hparams.n_ctx - tf.shape(attentions)[-1]]]
            p_attentions = tf.pad(attentions, paddings, 'CONSTANT', constant_values=0)

            r_attentions = tf.expand_dims(p_attentions, axis=1)

            attn_stack_array = tf.concat([attn_stack_array, r_attentions], axis=1)



            if top_p > 0.0:
                logits = top_p_logits(logits, p=top_p)
            else:
                logits = top_k_logits(logits, k=top_k)
            samples = tf.random.categorical(logits=logits, num_samples=1, dtype=tf.int32)
            return [
                tf.concat([past, next_outputs['presents']], axis=-2),
                tf.squeeze(samples, axis=[1]),
                tf.concat([output, samples], axis=1),
                attn_stack_array,
            ]

        def cond(*args):
            return True

        _, _, tokens, new_attn_stack = tf.nest.map_structure(
            tf.stop_gradient, 
            tf.while_loop(
                cond=cond, body=body,
                maximum_iterations=length,
                loop_vars=[
                    context_output['presents'],
                    context[:, -1],
                    context,
                    attentions_stack_array,
                ],
                shape_invariants=[
                    tf.TensorShape(model.past_shape(hparams=hparams, batch_size=batch_size)),
                    tf.TensorShape([batch_size]),
                    tf.TensorShape([batch_size, None]),
                    # tf.TensorShape([None, batch_size, hparams.n_layer, hparams.n_head, 1, hparams.n_ctx]),
                    tf.TensorShape([batch_size, None, hparams.n_layer, 1, hparams.n_ctx]),
                ]
            )
        )
        print(new_attn_stack.shape)

        return tokens, new_attn_stack

def attentions_from_given(*, hparams, context, length, batch_size=None, temperature=1, top_k=0, top_p=0.0):



    def step(hparams, tokens, past=None):
        print(f"step Tokens: {tokens}")
        lm_output = model.model(hparams=hparams, X=tokens, past=past, reuse=tf.compat.v1.AUTO_REUSE)

        print("TEST STEP")
        logits = lm_output['logits'][:, :, :hparams.n_vocab]
        presents = lm_output['present']
        presents.set_shape(model.past_shape(hparams=hparams, batch_size=batch_size))
        print("TS 2")
        return {
            'logits': logits,
            'presents': presents,
            'attentions': lm_output['attentions']
        }

    def compress_attentions(hparams, tensor):
        # LAYER SELECTION
        if hasattr(hparams, "LAYER_SELECTION") and hparams.LAYER_SELECTION is not None:
            tensor = tensor[:, :, hparams.LAYER_SELECTION:hparams.LAYER_SELECTION+1]
            # Ugly to preserve rank

        # HEAD SELECTION
        if hasattr(hparams, "HEAD_SELECTION") and hparams.HEAD_SELECTION is not None:
            tensor = tensor[:, :, :, hparams.HEAD_SELECTION:hparams.HEAD_SELECTION+1]
            # Ugly to preserve rank

        # LAYER REDUCTION
        if hasattr(hparams, "LAYER_REDUCTION") and hparams.LAYER_REDUCTION:
            tensor = tf.reduce_max(tensor, axis=2, keepdims=True)

        # HEAD REDUCTION
        if hasattr(hparams, "HEAD_REDUCTION") and hparams.HEAD_REDUCTION:
            tensor = tf.reduce_max(tensor, axis=3, keepdims=True)
        
        return tensor            


    with tf.compat.v1.name_scope('get_attentions'):
        # Don't feed the last context token -- leave that to the loop below
        # TODO: Would be slightly faster if we called step on the entire context,
        # rather than leaving the last token transformer calculation to the while loop.

        # attentions_stack_array = tf.zeros([0, batch_size, hparams.n_layer, hparams.n_head, ?sample?, hparams.n_ctx])
        attentions_stack_array = tf.zeros([batch_size, 0, hparams.n_layer,hparams.n_head, 1, hparams.n_ctx])
        print(f"ASA Shape: {attentions_stack_array.shape}")
        attentions_stack_array = compress_attentions(hparams, attentions_stack_array)
        print(f"ASA Shape: {attentions_stack_array.shape}")

        initial_output = step(hparams, context[:, :1])

        def body(past, attn_stack_array):

            i = tf.shape(attn_stack_array)[1]
            # i = 10
            next_outputs = step(hparams, context[:, i, tf.newaxis], past=past)
            # logits = next_outputs['logits'][:, -1, :]  / tf.cast(temperature, dtype=tf.float32)

            attentions = next_outputs['attentions']
            attentions = tf.expand_dims(attentions, axis=1)


            paddings = [[0,0], [0,0], [0,0], [0,0], [0,0], [0, hparams.n_ctx - tf.shape(attentions)[-1]]]
            attentions = tf.pad(attentions, paddings, 'CONSTANT', constant_values=0)

            attentions = compress_attentions(hparams, attentions)


            attn_stack_array = tf.concat([attn_stack_array, attentions], axis=1)

            return [
                tf.concat([past, next_outputs['presents']], axis=-2),
                attn_stack_array,
            ]


        def cond(*args):
            return True


        attention_invariant_shape = attentions_stack_array.shape.as_list()
        attention_invariant_shape[1] = None
        _, attentions_stack_array = tf.nest.map_structure(
            tf.stop_gradient, 
            tf.while_loop(
                cond=cond, body=body,
                maximum_iterations=length,
                loop_vars=[
                    initial_output['presents'],
                    attentions_stack_array,
                ],
                shape_invariants=[
                    tf.TensorShape(model.past_shape(hparams=hparams, batch_size=batch_size)),
                    tf.TensorShape(attention_invariant_shape)
                ]
            )
        )



        return None, attentions_stack_array