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

from skip_thoughts import configuration
from skip_thoughts import encoder_manager

# Add current path in search path as skip-thought is under this path
cur_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(cur_path)

sys.path.append('../utils')
from utils import CharWord2vec

logger = logging.getLogger(__name__)

class SkipThought(Text2vecBase):
    """Sentence encoder using Skip-thought

    Arguments:

    Properties:
        encoder: the sentence encoder model
        char_w2v: the word2vec model for out-of-vocab words

    """
    def __init__(self):
        super(SkipThought, self).__init__()

        cur_path = os.path.abspath(os.path.dirname(__file__))
        # Set paths to the model.
        VOCAB_FILE = os.path.join(cur_path, "../../models/skip_thoughts_uni_2017_02_02/vocab.txt")
        EMBEDDING_MATRIX_FILE = os.path.join(cur_path, "../../models/skip_thoughts_uni_2017_02_02/embeddings.npy")
        CHECKPOINT_PATH = os.path.join(cur_path, "../../models/skip_thoughts_uni_2017_02_02/model.ckpt-501424")
        PROJ_MODEL_PATH = os.path.join(cur_path, "../../models/linear_projection.m")

        self.encoder = encoder_manager.EncoderManager()
        self.encoder.load_model(configuration.model_config(),
                           vocabulary_file=VOCAB_FILE,
                           embedding_matrix_file=EMBEDDING_MATRIX_FILE,
                           checkpoint_path=CHECKPOINT_PATH)
        self.char_w2v = CharWord2vec(proj = PROJ_MODEL_PATH)



    def encode_sents(self, sents):
	"""Encode many sentence to vectors

	Arguments:
	    sents: sentences
	Returns:
	    list of 300-dims numpy array
      	"""
        encodings = self.encoder.encode(sents, self.char_w2v)
        return encodings

