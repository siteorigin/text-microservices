#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import logging
import numpy as np
from FifoCache import FifoCache

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
logging.root.setLevel(level=logging.INFO)

class Word2vec(object):
    """Word2vec loader and getter

    Arguments:
        cache_size: the size of vector cache

    Properties:
        idx2word: list of words in vocab(list)
        words: word set for fast search
        word2idx: inverse vocab(dict)
        vec_path: the path of vector file
        cache: a fifo cache for vectors
    """
    def __init__(self, cache_size=1000):
        cur_path = os.path.abspath(os.path.dirname(__file__))

        # load vocab into memory
        vocab_path = os.path.join(cur_path, "../../models/cbow/glove.840B.300d.vocab.txt")
        logger.info('loading %s...', vocab_path)
        self.idx2word = []
        with open(vocab_path) as f:
            self.idx2word = [w.strip().lower().decode('utf-8') for w in f.readlines()]
        self.word2idx = dict([(w,idx)for idx,w in enumerate(self.idx2word)])
        self.words = set(self.idx2word)
        logger.info("Load done. Vocab size: %d", len(self.idx2word))

        self.vec_path = os.path.join(cur_path, "../../models/cbow/glove.840B.300d.vector.txt")
        # make sure vector number equal vocab size
        with open(self.vec_path) as fp:
            num_lines = sum([1 for line in fp])
        if num_lines != len(self.idx2word):
            raise Exception('Line number of vocab and vector are not equal!')

        # init a cache for vectors
        self.cache = FifoCache(cache_size)

    def get_vec(self, w):
	"""Get vector of word w

	Arguments:
	    w: a word
	Returns:
	    word vector
      	"""

        # return a random vector if not found, otherwise convert to idx
        if w not in self.words:
            return np.random.rand(300)
        w = self.word2idx[w.strip().lower().decode('utf-8')]

        # find in cache firstly
        if self.cache.has_key(w):
            return self.cache[w]

        # not hit in cache, find in file
        vec = None
        with open(self.vec_path) as fp:
            for i, line in enumerate(fp):
                if i == w:
                    vec = np.array([float(num) for num in line.split(' ')])
        if vec is None or len(vec) != 300:
            vec = np.random.rand(300)

        # put into cache and return
        self.cache[w] = vec
        return vec

if __name__=='__main__':
    w2v = Word2vec(cache_size=100)
    for x in xrange(10000):
        vec = w2v.get_vec(str(x))
