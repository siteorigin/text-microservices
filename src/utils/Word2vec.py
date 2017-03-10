#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import logging
import numpy as np
from collections import defaultdict
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

    def get_vec(self, sent):
	"""Get vectors of words in sentence

	Arguments:
	    sentence: a list of word(lower case)
	Returns:
	    word vector
      	"""

        result = [None] * len(sent)

        # return a random vector if not found, otherwise convert to idx
        w_not_hit = defaultdict(list) # there may be repeat words, so we use list
        for idx,w in enumerate(sent):
            try:
                w = w.decode('utf-8')
            except:
                pass

            if w not in self.words:
                # logger.debug("%s not in vocab, generate random vector", w)
                result[idx] = np.random.rand(300)
            else:
                w = self.word2idx[w] # convert word to idx now
                # find in cache firstly, record idx if not found
                if self.cache.has_key(w):
                    logger.debug("%s: hit in cache", self.idx2word[w])
                    result[idx] = self.cache[w]
                else:
                    logger.debug("%s: not in cache", self.idx2word[w])
                    w_not_hit[w].append(idx)

        # some words not hit in cache, find in file
        if len(w_not_hit) > 0:
            with open(self.vec_path) as fp:
                for line_num, line in enumerate(fp):
                    if line_num in w_not_hit.keys():
                        vec = np.array([float(num) for num in line.split(' ')])
                        self.cache[line_num] = vec # put into cache
                        for idx in w_not_hit[line_num]: # put into result
                            result[idx] = vec
                        w_not_hit.pop(line_num) # remove from candidate
                        if len(w_not_hit) == 0: # stop search if all vector found
                            break
        return result

if __name__=='__main__':
    w2v = Word2vec(cache_size=11)
    for x in xrange(100):
        vec = w2v.get_vec([str(num) for num in list(range(x, x+10))])