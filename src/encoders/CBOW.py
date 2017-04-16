#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import pickle
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

    Arguments:
        use_char: whether using character word embedding for unknown words

    Properties:
        w2v: Word2vec class
        char_w2v: the word2vec model for out-of-vocab words
        use_char: whether using character word embedding for unknown words
    """
    def __init__(self, use_char=True):
        super(CBOW, self).__init__()
        self.use_char = use_char

        cur_path = os.path.abspath(os.path.dirname(__file__))

        self.w2v = Word2vec(cache_size=1000)

        if self.use_char:
            self.char_w2v = CharWord2vec()
            PROJ_MODEL_PATH = os.path.join(cur_path, "../../models/char_word2vec/cbow_linear_projection.m")
            with open(PROJ_MODEL_PATH) as f:
                self.proj = pickle.load(f)

    def encode_sent(self, raw_sent):
	"""Encode a sentence to vector

	Arguments:
	    raw_sent: a sentence
	Returns:
	    300-dims numpy array
      	"""
        sent = word_tokenize(raw_sent)
        sent_vec = self.w2v.get_vec(sent)

        # sent_vec = [vec for vec in sent_vec if vec is not None]
        # handle out-of-vocab word
        unk_word = [w for v,w in zip(sent_vec, sent) if v is None]
        if self.use_char and len(unk_word) > 0 :
            unk_word_vec = self.char_w2v.get_vec(unk_word)
            # project the char-level embedding to specific vector space
            unk_word_vec = self.proj.predict(unk_word_vec)
            jdx = 0
            for idx, vec in enumerate(sent_vec):
                if vec is None:
                    sent_vec[idx] = unk_word_vec[jdx]
                    jdx += 1
        else:
            sent_vec = [vec for vec in sent_vec if vec is not None]

        return np.mean(sent_vec, axis=0)

