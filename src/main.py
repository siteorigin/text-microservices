#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import json
import time
import hashlib
import logging
import numpy as np
from flask import Flask
from flask import request

from encoders import CBOW
from encoders import SkipThought

app = Flask(__name__)

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s: %(message)s')
logging.root.setLevel(level=logging.INFO)

cbow_glove_model = CBOW(use_char=True)
skip_thought_model = SkipThought()

@app.route("/_ah/health")
def check():
    return 'aha'

def auth():
    AUTH_SALT = 'aaabbbccc';
    if request.method == 'POST':
        data = request.get_json()
    else:
        data = request.args
    if data.has_key('user_email') and data.has_key('key_expire') and data.has_key('key'):
        user_email = data['user_email']
        key_expire = data['key_expire']
        key = data['key']
        if len(user_email) == 0 or int(time.time()) > int(key_expire):
            return False
        elif key == hashlib.sha1(AUTH_SALT + user_email + key_expire).hexdigest():
            return True
        else:
            return False
    else:
        return False

@app.route("/", methods=['GET', 'POST'])
def main():
    if os.environ.has_key('SONAR_AUTH_REQUESTS') and not auth():
        response = {'status': -1, 'msg': 'Login fail.'}
        return json.dumps(response)

    if request.method == 'POST':
        data = request.get_json()
    else:
        data = request.args
    req_text = data.get('text', '')
    req_type = data.get('type', 'text')
    req_model = data.get('model', 'cbow-glove')

    if len(req_text.strip()) == 0:
        response = {'status': 1, 'msg': 'Request text is empty.'}
        return json.dumps(response)

    response = {}
    model = None
    if req_model == 'cbow-glove':
        model = cbow_glove_model
    if req_model == 'skip-thought':
        model = skip_thought_model
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
    app.run(host='0.0.0.0')
