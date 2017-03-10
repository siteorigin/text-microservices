#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import logging
import numpy as np

logger = logging.getLogger('process_glove')
logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
logging.root.setLevel(level=logging.INFO)

def split_words_vectors():
    # split glove into two files: words and vectors
    cur_path = os.path.abspath(os.path.dirname(__file__))
    model_path = os.path.join(cur_path, "../models/cbow/glove.840B.300d.txt")
    logger.info('loading %s, this will take a while...' % model_path)
    with open(model_path) as f:
        data = f.readlines()
    idx2word = []
    vectors = []
    for line in data:
        line = line.split(' ')
        if len(line) != 301:
            continue
        idx2word.append(line[0].decode('utf-8').lower())
        vectors.append([float(x) for x in line[1:]])

    logger.info("Load done. Vocab size: %d", len(idx2word))
    vectors = np.asarray(vectors)

    '''
    logger.info("Normalize vectors, this will take long time...")
    norms = np.linalg.norm(vectors, axis=1).reshape((-1, 1))
    vectors = vectors/norms
    '''

    # save into two files
    vocab_path = os.path.join(cur_path, "../models/cbow/glove.840B.300d.vocab.txt")
    with open(vocab_path, 'wb') as f:
        f.write('\n'.join(idx2word).encode('utf-8'))
    vec_path = os.path.join(cur_path, "../models/cbow/glove.840B.300d.vector.txt")
    with open(vec_path, 'wb') as f:
        vectors = [[str(x) for x in v] for v in vectors]
        vectors = [' '.join(v).encode('utf-8') for v in vectors]
        f.write('\n'.join(vectors))

if __name__=='__main__':
    split_words_vectors()
