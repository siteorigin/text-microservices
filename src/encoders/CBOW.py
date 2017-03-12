#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import logging
import numpy as np
from nltk import word_tokenize

sys.path.append('../utils')
from utils import Word2vec
from Text2vecBase import Text2vecBase

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
logging.root.setLevel(level=logging.INFO)

class CBOW(Text2vecBase):
    """Basic sentence encoder using continuous bag-of-words.

    Properties:
        w2v: Word2vec class
    """
    def __init__(self):
        super(CBOW, self).__init__()
        self.w2v = Word2vec(cache_size=1000)

    def encode_sent(self, raw_sent):
	"""Encode a sentence to vector

	Arguments:
	    raw_sent: a sentence
	Returns:
	    300-dims numpy array
      	"""
        sent = word_tokenize(raw_sent.lower())
        sent_vec = self.w2v.get_vec(sent)
        sent_vec = [vec for vec in sent_vec if vec is not None]
        if len(sent_vec) == 0:
            return None
        else:
            return np.mean(sent_vec, axis=0)

