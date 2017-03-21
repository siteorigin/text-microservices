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
    model = skip_thought_model
    ff = open('../data/Crowdflower_pred.csv', 'wb')
    writer = csv.writer(ff, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    with open('../data/Crowdflower.csv', 'rb') as f:
        data = csv.reader(f, delimiter=',', quotechar='"')
        for idx,line in enumerate(data):
            if idx==0:
                line.append('score')
                writer.writerow(line)
                continue
            try:
                v1,_,_,_ = model.encode(line[1])
                v2,_,_,_ = model.encode(line[2])
                cos = np.dot(v1, v2) / np.sqrt(np.dot(v1, v1) * np.dot(v2, v2))
                line.append(cos)
                writer.writerow(line)
                # print line[4], cos
            except:
                print line[1], line[2]
    ff.close()


if __name__ == '__main__':
    # test_cbow()
    test_skip_thought()
