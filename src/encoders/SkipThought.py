#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import dill
import logging
import numpy as np
import tensorflow as tf
from nltk import word_tokenize

from Text2vecBase import Text2vecBase

# Add current path in search path as skip-thought is under this path
cur_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(cur_path)

from skipthought import SkipthoughtModel
from skipthought.data_utils import TextData

logger = logging.getLogger(__name__)

class SkipThought(Text2vecBase):
    """Sentence encoder using Skip-thought

    Properties:

    """
    def __init__(self):
        super(SkipThought, self).__init__()

        self.kwargs = {}
        self.kwargs['init_from'] = os.path.join(cur_path, "../../models/skip-thought/")

        logger.info("Check if I can restore model from {0}".format(self.kwargs['init_from']))
        # check if all necessary files exist
        assert os.path.isdir(self.kwargs['init_from']), "%s must be a a path" % self.kwargs['init_from']
        assert os.path.isfile(os.path.join(self.kwargs['init_from'], "config.pkl")), "config.pkl file does not exist in path %s" % self.kwargs['init_from']
        ckpt = tf.train.get_checkpoint_state(self.kwargs['init_from'])
        assert ckpt, "No checkpoint found"
        assert ckpt.model_checkpoint_path, "No model path found in checkpoint"

        # open old config and check if models are compatible
        with open(os.path.join(self.kwargs['init_from'], 'config.pkl'), 'rb') as f:
            saved_model_args = dill.load(f)
            saved_model_args.update(self.kwargs)
            self.kwargs = saved_model_args
            logger.info("Args load done.")
            logger.info(self.kwargs)

        self.vocab = TextData.load(os.path.join(self.kwargs['init_from'], 'vocab.pkl'))
        vocab_size = len(self.vocab)
        logger.info("actual vocab_size={0}".format(vocab_size))

        self.sess = tf.InteractiveSession()
        self.model = SkipthoughtModel(self.kwargs['cell_type'], self.kwargs['num_hidden'], self.kwargs['num_layers'],
                                 self.kwargs['embedding_size'], vocab_size, self.kwargs['learning_rate'],
                                 self.kwargs['decay_rate'], 0, self.kwargs['grad_clip'],
                                 self.kwargs['num_samples'], self.kwargs['max_len'], only_encoder=True)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=20)
        saver.restore(self.sess, ckpt.model_checkpoint_path)
        print("Restored from {0}".format(ckpt.model_checkpoint_path))

    def encode_sents(self, sents):
	"""Encode many sentence to vectors

	Arguments:
	    sents: sentences
	Returns:
	    list of 300-dims numpy array
      	"""
        textdata = TextData(sents, max_len=self.kwargs['max_len'], max_vocab_size=self.kwargs['max_vocab_size'], vocab=self.vocab)
        it = textdata.lines_data_iterator(textdata.dataset, textdata.max_len, self.kwargs['batch_size'], shuffle=False)
        vectors = None
        for b, batch in enumerate(it):
            encoder_state, feed_dict = self.model.encode_step(batch)
            vec = self.sess.run(encoder_state, feed_dict=feed_dict)
            if vectors is None:
                vectors = vec
            else:
                vectors = np.concatenate((vectors, vec), axis=0)
        return vectors

