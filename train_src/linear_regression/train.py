
import pickle
import numpy as np
from sklearn.linear_model import LinearRegression

vocab_file = '../../models/skip_thoughts_uni_2017_02_02/vocab.txt'
source_vec_file = '../../data/char_embedding.npy'
target_vec_file = '../../models/skip_thoughts_uni_2017_02_02/embeddings.npy'

def vec_sim(v1, v2):
    cos = np.dot(v1, v2) / np.sqrt(np.dot(v1, v1) * np.dot(v2, v2))
    return (cos + 1) / 2

def train():
    with open(vocab_file) as f:
        vocab = f.readlines()
        vocab = [w.strip().lower() for w in vocab]
    print len(vocab)

    source_vec = np.load(source_vec_file)
    print source_vec.shape
    # assert len(vocab) == source_vec.shape[0]
    target_vec = np.load(target_vec_file)[:500, :]
    print target_vec.shape
    # assert len(vocab) == target_vec.shape[0]

    proj_model = LinearRegression(n_jobs=4)
    proj_model.fit(source_vec, target_vec)
    with open('../../models/linear_projection.m', 'w') as f:
        pickle.dump(proj_model, f)

def test(test_idx = 88):
    with open(vocab_file) as f:
        vocab = f.readlines()
        vocab = [w.strip().lower() for w in vocab][:500]
    print len(vocab)

    source_vec = np.load(source_vec_file)
    print source_vec.shape

    with open('../../models/linear_projection.m') as f:
        proj_model = pickle.load(f)

    # test similarity on a word
    result = proj_model.predict(source_vec)
    tups = []
    for w,vec in zip(vocab, result):
        tups.append((vocab[test_idx], w, vec_sim(result[test_idx, :], vec)))
    tups = sorted(tups, key=lambda x: x[2], reverse=True)
    for tup in tups[:10]:
        print("%s\t%s\t%f" % tup)

if __name__ == '__main__':
    train()
    test(111)
