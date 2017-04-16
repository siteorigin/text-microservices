
import os
import numpy as np

model_path = '../models/skip_thoughts_uni_2017_02_02'

m = np.load(os.path.join(model_path, 'embeddings.npy'))
vocab = [w.strip() for w in open(os.path.join(model_path, 'vocab.txt')).readlines()]

f = open(os.path.join(model_path, 'embeddings.txt'), 'w')
for w,v in zip(vocab, m):
    v = ' '.join(['%.4f'%float(num) for num in v])
    f.write(w + ' ' + v + '\n')
f.close()
