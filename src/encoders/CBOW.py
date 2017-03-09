#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import logging
import numpy as np
from nltk import word_tokenize

logger = logging.getLogger('cbow')
logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
logging.root.setLevel(level=logging.INFO)

class CBOW():
    """Basic sentence encoder using continuous bag-of-words.

    Properties:
        idx2word: list of words in vocab(list)
        words: word set for fast search
        word2idx: inverse vocab(dict)
        vectors: word vectors(2-D array, each row is a 300-dim vector of a word)
    """
    def __init__(self):
        cur_path = os.path.abspath(os.path.dirname(__file__))
        model_path = os.path.join(cur_path, "../../models/cbow/glove.840B.300d.txt")
        logger.info('loading %s, this will take a while...', model_path)
        self.idx2word = []
        self.vectors = []
        with open(model_path) as f:
            data = f.readlines()
        for line in data:
            line = line.split(' ')
            if len(line) != 301:
                continue
            self.idx2word.append(line[0].decode('utf-8').lower())
            self.vectors.append([float(x) for x in line[1:]])
            if len(self.idx2word) % 100000 == 0:
                logger.info('%d words loaded', len(self.idx2word))
        self.word2idx = dict([(w,idx)for idx,w in enumerate(self.idx2word)])
        self.words = set(self.idx2word)
        logger.info("Load done. Vocab size: %d", len(self.idx2word))
        self.vectors = np.asarray(self.vectors)
        # normalize vectors
        norms = np.linalg.norm(self.vectors, axis=1).reshape((-1, 1))
        self.vectors = self.vectors/norms

    def encode(self, raw_sent):
	"""Encode a sentence to vector

	Arguments:
	    raw_sent: a sentence
	Returns:
	    300-dims numpy array
      	"""
        sent = word_tokenize(raw_sent.lower())
        sent = [self.word2idx[w] for w in sent if w in self.words]
        sent_vec = self.vectors[sent]
        if len(sent_vec) == 0:
            logger.info('All word not found in vocab, return random vector')
            return np.random.rand(300)
        else:
            return np.mean(sent_vec, axis=0)

