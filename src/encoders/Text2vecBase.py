#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import logging
import numpy as np
from nltk.tokenize import sent_tokenize

sys.path.append('../utils')
from utils import Word2vec
from utils.helper import vec_sim

logger = logging.getLogger(__name__)

class Text2vecBase(object):
    """A base class for Text2vec
    This class split the document into sentences firstly.
    Then encode each sentence, compute the centroid vector.
    Compute the salience of each sentence according to its distance to the centroid vector.
    Finally return the weighted average vector as the document vector.

    """
    def __init__(self):
        pass

    def encode(self, doc):
	"""Encode a document to vector

	Arguments:
	    document: a document
	Returns:
            doc_vec: a feature vector for the entire document
            sents: a list of the sentences of the document
            features: a list of the vectors of each sentence
            salience: alist of the importance of each sentence

      	"""
        sents = sent_tokenize(doc.strip())
        sents_vec = self.encode_sents(sents)
        sents_vec_clean = [vec for vec in sents_vec if vec is not None]
        if len(sents_vec_clean) == 0:
            return None, None, None, None
        else:
            sents_mean = np.mean(sents_vec_clean, axis=0)
            sents_sim = [vec_sim(vec, sents_mean) if vec is not None else None for vec in sents_vec]
            sents_sim_clean = [vec for vec in sents_sim if vec is not None]
            doc_vec = np.average(sents_vec_clean, axis=0, weights=sents_sim_clean)
            return doc_vec, sents, sents_vec, sents_sim

    def encode_sents(self, sents):
        """Encode many sentences to vectors
        This is a base class for sentence encoders, So we don't implement the method here.
        we simply return a random vector.

	Arguments:
	    sents: sentences
	Returns:
	    list of 300-dims numpy array
      	"""
        return [self.encode_sent(sent) for sent in sents]

    def encode_sent(self, raw_sent):
        """Encode a sentence to vector
        This is a base class for sentence encoders, So we don't implement the method here.
        we simply return a random vector.

	Arguments:
	    raw_sent: a sentence
	Returns:
	    300-dims numpy array
      	"""
        return np.random(300)
