#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import json
import logging
import numpy as np
from flask import Flask
from flask import request

from encoders import CBOW

app = Flask(__name__)

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
logging.root.setLevel(level=logging.DEBUG)

cbow_glove_model = CBOW()

def test():
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
    logger.debug("End testing CBOW-Glove model.")

@app.route("/", methods=['GET', 'POST'])
def main():
    if request.method == 'POST':
        req_text = request.values.get('text', '')
        req_type = request.values.get('type', 'text')
        req_model = request.values.get('model', 'cbow-glove')
    else:
        req_text = request.args.get('text', '')
        req_type = request.args.get('type', 'text')
        req_model = request.args.get('model', 'cbow-glove')


    response = {}
    model = None
    if req_model == 'cbow-glove':
        model = cbow_glove_model
    else:
        response = {'status': 1, 'msg': 'Model not found.'}

    if model is not None:
        vector, sents, sents_vec, sents_sim = model.encode(req_text)
        if vector is None:
            response = {'status': 1}
        else:
            response = {'status': 0}
            response['features'] = [float(x) for x in vector]
            response['sentences'] = []
            for sent, sent_vec, sent_sim in zip(sents, sents_vec, sents_sim):
                if sent_vec is None:
                    tmp_res = {'status': 1}
                    tmp_res['text'] = sent
                else:
                    tmp_res = {'status': 0}
                    tmp_res['text'] = sent
                    tmp_res['features'] = [float(x) for x in sent_vec]
                    tmp_res['salience'] = sent_sim
                response['sentences'].append(tmp_res)
    return json.dumps(response)

if __name__ == '__main__':
    logger.info("Running %s" % ' '.join(sys.argv))
    test()
    app.run()
