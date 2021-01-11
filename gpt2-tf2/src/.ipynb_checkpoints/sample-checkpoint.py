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

        attentions = lm_output['attentions']
        logits = lm_output['logits'][:, :, :hparams.n_vocab]
        presents = lm_output['present']
        presents.set_shape(model.past_shape(hparams=hparams, batch_size=batch_size))
        return {
            'logits': logits,
            'presents': presents,
            'attentions': attentions,
        }

    with tf.compat.v1.name_scope('sample_sequence'):
        # Don't feed the last context token -- leave that to the loop below
        # TODO: Would be slightly faster if we called step on the entire context,
        # rather than leaving the last token transformer calculation to the while loop.
        context_output = step(hparams, context[:, :-1])
        whole_attention = tf.zeros([batch_size, 0, hparams.n_layer, hparams.n_head,1, 2])

        def body(past, prev, output, whole_attention):
            next_outputs = step(hparams, prev[:, tf.newaxis], past=past)
            logits = next_outputs['logits'][:, -1, :]  / tf.cast(temperature, dtype=tf.float32)
            if top_p > 0.0:
                logits = top_p_logits(logits, p=top_p)
            else:
                logits = top_k_logits(logits, k=top_k)
            samples = tf.random.categorical(logits=logits, num_samples=1, dtype=tf.int32)
            return [
                tf.concat([past, next_outputs['presents']], axis=-2),
                tf.squeeze(samples, axis=[1]),
                tf.concat([output, samples], axis=1),
                tf.concat([whole_attention, tf.reshape(next_outputs['attentions'], (batch_size, 1, hparams.n_layer, hparams.n_head, 1, -1))], axis=1),
            ]

        def cond(*args):
            return True

        _, _, tokens, attentions = tf.nest.map_structure(
            tf.stop_gradient, 
            tf.while_loop(
                cond=cond, body=body,
                maximum_iterations=length,
                loop_vars=[
                    context_output['presents'],
                    context[:, -1],
                    context,
                    whole_attention,
                ],
                shape_invariants=[
                    tf.TensorShape(model.past_shape(hparams=hparams, batch_size=batch_size)),
                    tf.TensorShape([batch_size]),
                    tf.TensorShape([batch_size, None]),
                    tf.TensorShape([batch_size, None, hparams.n_layer, hparams.n_head,1, 2])
                ]
            )
        )

        return tokens, attentions
