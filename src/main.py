#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import json
import logging
import numpy as np
import tornado.web
import tornado.ioloop

from encoders import CBOW

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
    v1 = model.encode(sent1)
    v2 = model.encode(sent2)
    cos = np.dot(v1, v2) / np.sqrt(np.dot(v1, v1) * np.dot(v2, v2))
    logger.debug("Cosine: %f", cos)
    logger.debug("End testing CBOW-Glove model.")

class MainHandler(tornado.web.RequestHandler):
    def get(self):
        self.post()

    def post(self):
        req_text = self.get_argument('text', 'hello world')
        req_type = self.get_argument('type', 'text')
        req_model = self.get_argument('model', 'cbow-glove')
        if req_model == 'cbow-glove':
            model = cbow_glove_model
        else:
            model = cbow_glove_model
        vector = [float(x) for x in model.encode(req_text)]
        self.write(json.dumps(vector))

if __name__ == '__main__':
    logger.info("Running %s" % ' '.join(sys.argv))

    test()

    port = 8888
    logger.info("Listening on port %s" % port)
    app = tornado.web.Application([
        (r"/", MainHandler),
        ])
    app.listen(port)
    tornado.ioloop.IOLoop.current().start()


