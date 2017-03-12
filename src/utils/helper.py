#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

def vec_sim(v1, v2):
    cos = np.dot(v1, v2) / np.sqrt(np.dot(v1, v1) * np.dot(v2, v2))
    return (cos + 1) / 2
