#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv
import numpy as np
from encoders import CBOW
from encoders import SkipThought

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s: %(message)s')
logging.root.setLevel(level=logging.DEBUG)

# cbow_glove_model = CBOW()
skip_thought_model = SkipThought()

def test_cbow():
    logger.debug("Testing CBOW... ")
    sent1 = 'hotels in new york'
    sent2 = 'restaurants in New York'
    logger.debug("Test sentence1: %s", sent1)
    logger.debug("Test sentence2: %s", sent2)
    logger.debug("Start testing CBOW-Glove model...")
    model = cbow_glove_model
    v1,_,_,_ = model.encode(sent1)
    v2,_,_,_ = model.encode(sent2)
    cos = np.dot(v1, v2) / np.sqrt(np.dot(v1, v1) * np.dot(v2, v2))
    logger.debug("Cosine: %f", cos)
    logger.debug("End testing CBOW-Glove model. \n")

def test_skip_thought():
    logger.debug("Testing CBOW... ")
    sent1 = 'hotels in New York'
    sent2 = 'restaurants in New York'
    logger.debug("Test sentence1: %s", sent1)
    logger.debug("Test sentence2: %s", sent2)
    logger.debug("Start testing skip-thought model...")
    model = skip_thought_model
    v1,_,_,_ = model.encode(sent1)
    v2,_,_,_ = model.encode(sent2)
    cos = np.dot(v1, v2) / np.sqrt(np.dot(v1, v1) * np.dot(v2, v2))
    logger.debug("Cosine: %f", cos)
    logger.debug("End testing skip-thought model. \n")

if __name__ == '__main__':
    # test_cbow()
    test_skip_thought()
