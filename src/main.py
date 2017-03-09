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

cbow_glove_model = CBOW()

def test():
    model = cbow_glove_model
    v1 = model.encode('hotels in New York')
    v2 = model.encode('restaurants in New York')
    print np.dot(v1, v2) / np.sqrt(np.dot(v1, v1) * np.dot(v2, v2))

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
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("Running %s" % ' '.join(sys.argv))

    app = tornado.web.Application([
        (r"/", MainHandler),
        ])
    app.listen(8888)
    tornado.ioloop.IOLoop.current().start()


