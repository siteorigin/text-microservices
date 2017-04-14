#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import inspect
import logging
import pickle
import numpy as np
import char_w2v_model
import tensorflow as tf
from collections import defaultdict
from char_w2v_data_reader import sents_to_batch, Vocab

logger = logging.getLogger(__name__)
flags = tf.flags

# data
flags.DEFINE_string('data_dir',    'data',   'data directory. Should contain train.txt/valid.txt/test.txt with input data')
flags.DEFINE_string('train_dir',   'cv',     'training directory (models and summaries are saved there periodically)')
flags.DEFINE_string('load_model',   None,    '(optional) filename of the model to load. Useful for re-starting training from a checkpoint')

# model params
flags.DEFINE_integer('rnn_size',        650,                            'size of LSTM internal state')
flags.DEFINE_integer('highway_layers',  2,                              'number of highway layers')
flags.DEFINE_integer('char_embed_size', 15,                             'dimensionality of character embeddings')
flags.DEFINE_string ('kernels',         '[1,2,3,4,5,6,7]',              'CNN kernel widths')
flags.DEFINE_string ('kernel_features', '[50,100,150,200,200,200,200]', 'number of features in the CNN kernel')
flags.DEFINE_integer('rnn_layers',      2,                              'number of layers in the LSTM')
flags.DEFINE_float  ('dropout',         0.5,                            'dropout. 0 = no dropout')

# optimization
flags.DEFINE_integer('num_unroll_steps',    35,   'number of timesteps to unroll for')
flags.DEFINE_integer('batch_size',          1,   'number of sequences to train on in parallel')
flags.DEFINE_integer('max_word_length',     65,   'maximum word length')

# bookkeeping
flags.DEFINE_integer('seed',           3435, 'random number generator seed')
flags.DEFINE_string ('EOS',            '+',  '<EOS> symbol. should be a single unused character (like +) for PTB and blank for others')

FLAGS = flags.FLAGS

'''
Python singleton with parameters (so the same parameters get you the same object)
with support to default arguments and passing arguments as kwargs (but no support for pure kwargs).
'''
class Singleton(type):
    _instances = {}
    _init = {}

    def __init__(cls, name, bases, dct):
        cls._init[cls] = dct.get('__init__', None)

    def __call__(cls, *args, **kwargs):
        init = cls._init[cls]
        if init is not None:
            key = (cls, frozenset(
                    inspect.getcallargs(init, None, *args, **kwargs).items()))
        else:
            key = cls

        if key not in cls._instances:
            cls._instances[key] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[key]

class CharWord2vec(object):
    """Character-level Word2vec

    Arguments:
        proj: the model path to project output vector to a new vector space

    Properties:
        char_vocab: vocab of characters
        word_vocab: vocab of output words (Only the size is used to init model)
        session: the tensorflow session
        m: the model for char-level word2vec
        proj: the model for project the char-level word2vec to a new vector space
    """
    __metaclass__ = Singleton
    def __init__(self):
        cur_path = os.path.abspath(os.path.dirname(__file__))
        ckp_path = os.path.join(cur_path, "../../models/char_word2vec")
        ckp_file = os.path.join(ckp_path, "epoch024_4.4100.model")
        self.char_vocab =  Vocab.load(os.path.join(ckp_path, 'char_vocab.pkl'))
        self.word_vocab =  Vocab.load(os.path.join(ckp_path, 'word_vocab.pkl'))

        with tf.Graph().as_default():
            self.session = tf.Session()
            with self.session.as_default():
                ''' build inference graph '''
                with tf.variable_scope("Model"):
                    self.m = char_w2v_model.inference_graph(
                            char_vocab_size=self.char_vocab.size,
                            word_vocab_size=self.word_vocab.size,
                            char_embed_size=FLAGS.char_embed_size,
                            batch_size=FLAGS.batch_size,
                            num_highway_layers=FLAGS.highway_layers,
                            num_rnn_layers=FLAGS.rnn_layers,
                            rnn_size=FLAGS.rnn_size,
                            max_word_length=FLAGS.max_word_length,
                            kernels=eval(FLAGS.kernels),
                            kernel_features=eval(FLAGS.kernel_features),
                            num_unroll_steps=FLAGS.num_unroll_steps,
                            dropout=0)

                    saver = tf.train.Saver()
                    saver.restore(self.session, ckp_file)

                    logger.info('Loaded model from %s', ckp_file)
        del self.word_vocab


    def get_vec(self, words):
	"""Get vectors of words in sentence

	Arguments:
	    words: a list of words
	Returns:
	    2-D array, each line is the vector of the each word
      	"""
        char_tensors = sents_to_batch(words, FLAGS.max_word_length, self.char_vocab)
        word_num = char_tensors.shape[0]

        # padding tensors to match shape of model input
        if word_num % FLAGS.num_unroll_steps > 0:
            num_need = FLAGS.num_unroll_steps-word_num%FLAGS.num_unroll_steps
            for i in xrange(num_need):
                char_tensors = np.concatenate((char_tensors, char_tensors[:1,:]))

        # run batch by batch
        result = []
        for i in xrange(int(char_tensors.shape[0] / FLAGS.num_unroll_steps)):
            batch = char_tensors[i*FLAGS.num_unroll_steps: (i+1)*FLAGS.num_unroll_steps, :]
            batch = np.expand_dims(batch, axis=0)
            vectors = self.session.run(self.m.input_cnn, {self.m.input: batch})
            vectors = vectors.reshape(vectors.shape[1:])
            result.append(vectors)
        result = np.concatenate(result, axis=0)
        # cut off the line we append
        result = result[:word_num, :]

        return result

if __name__=='__main__':
    from helper import vec_sim
    w2v = CharWord2vec()

    def test_words(words):
        vec = w2v.get_vec(words)
        print(vec.shape)
        result = []
        for w,v in zip(words, vec):
            result.append((words[0],w,vec_sim(vec[0,:], v)))
        result = sorted(result, key=lambda x: x[2], reverse=True)
        for x in result:
            print('%s\t%s\t%f' % x)

    words = ['monday', 'fridays', 'mondays', 'minday', 'day', 'test', 'apple', 'google', 'test']
    test_words(words)

    words = ['apple', 'fridays', 'mondays', 'minday', 'day', 'test', 'banana', 'orange', 'google', 'apples']
    test_words(words)
