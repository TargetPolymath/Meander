#!/usr/bin/env python3

import fire
import json
import os
import numpy as np
import tensorflow as tf

import model, sample, encoder

def get_attentions(
    analysis_text,
    model_name='117M',
    seed=None,
    temperature=1,
    top_k=0,
    top_p=0.0,
):
    
    

    enc = encoder.get_encoder(model_name)
    hparams = model.default_hparams()
    with open(os.path.join('models', model_name, 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))



    with tf.compat.v1.Session(graph=tf.Graph()) as sess:

        context_tokens = enc.encode(analysis_text)
        context_redecoded = enc.raw_decode(context_tokens)
        print(context_tokens)
        print("Test")

        length = len(context_tokens)
        if length > hparams.n_ctx:
            raise ValueError("Can't get samples longer than window size: %s" % hparams.n_ctx)

        # hparams.add_hparam("LAYER_SELECTION", 0)

        context = tf.compat.v1.placeholder(tf.int32, [1, None])
        np.random.seed(seed)
        tf.compat.v1.set_random_seed(seed)
        
        _, attn_stack = sample.attentions_from_given(
            hparams=hparams, length=length,
            context=context,
            batch_size=1,
            temperature=temperature, top_k=top_k, top_p=top_p
        )

        # output = pre_output[:, 1:]

        saver = tf.compat.v1.train.Saver()
        ckpt = tf.train.latest_checkpoint(os.path.join('models', model_name))
        saver.restore(sess, ckpt)


        sample_data = [[]]

        attns = sess.run(attn_stack, feed_dict={
            context: [context_tokens]
        })

        sample_data[0].append(list(zip(context_redecoded, attns[0])))
        # text = enc.decode(out[0])
        # print("=" * 40 + " SAMPLE " + str(generated) + " " + "=" * 40)
        # print(text)


        print(f"Sample Data: {len(sample_data[0])}")
        return sample_data
            


if __name__ == '__main__':
    fire.Fire(interact_model)
 