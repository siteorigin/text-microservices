

import numpy as np
m = np.load('embeddings.npy')
vocab = [w.strip() for w in open('vocab.txt').readlines()]
for w,v in zip(vocab, m):
f = open('embeddings.txt', 'w')
for w,v in zip(vocab, m):
    v = ' '.join(['%.4f'%float(num) for num in v])
    f.write(w + ' ' + v + '\n')
f.close()
