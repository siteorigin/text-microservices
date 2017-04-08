#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import logging
import numpy as np
from nltk import word_tokenize

sys.path.append('../utils')
from utils import Word2vec
from utils import CharWord2vec
from Text2vecBase import Text2vecBase

logger = logging.getLogger(__name__)

class CBOW(Text2vecBase):
    """Basic sentence encoder using continuous bag-of-words.

    Properties:
        w2v: Word2vec class
        char_w2v: the word2vec model for out-of-vocab words
    """
    def __init__(self):
        super(CBOW, self).__init__()

        cur_path = os.path.abspath(os.path.dirname(__file__))
        PROJ_MODEL_PATH = os.path.join(cur_path, "../../models/char_word2vec/cbow-glove_linear_projection.m")

        self.w2v = Word2vec(cache_size=1000)
        self.char_w2v = CharWord2vec(proj = PROJ_MODEL_PATH)

    def encode_sent(self, raw_sent):
	"""Encode a sentence to vector

	Arguments:
	    raw_sent: a sentence
	Returns:
	    300-dims numpy array
      	"""
        sent = word_tokenize(raw_sent.lower())
        sent_vec = self.w2v.get_vec(sent)

        # sent_vec = [vec for vec in sent_vec if vec is not None]
        # handle out-of-vocab word
        unk_word = [w for v,w in zip(sent_vec, sent) if v is None]
        if len(unk_word) > 0 :
            unk_word_vec = self.char_w2v.get_vec(unk_word)
            jdx = 0
            for idx, vec in enumerate(sent_vec):
                if vec is None:
                    sent_vec[idx] = unk_word_vec[jdx]
                    jdx += 1

        return np.mean(sent_vec, axis=0)

