
import pickle
import numpy as np
from sklearn.linear_model import LinearRegression

source_vec_file = '../../data/skip-thought_char_embedding.npy'
vocab_file = '../../models/skip_thoughts_uni_2017_02_02/vocab.txt'
target_vec_file = '../../models/skip_thoughts_uni_2017_02_02/embeddings.npy'
save_model_file = '../../models/skip-thought_linear_projection.m'

'''
source_vec_file = '../../data/cbow_char_embedding.npy'
vocab_file = '../../models/cbow/glove.840B.300d.vocab.txt'
target_vec_file = '../../models/cbow/glove.840B.300d.txt'
save_model_file = '../../models/cbow_linear_projection.m'
'''

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
    if target_vec_file.endswith('npy'):
        target_vec = np.load(target_vec_file)
    else:
        target_vec = None
        with open(target_vec_file) as f:
            for line in f:
                line = line.strip().split(' ')
                if len(line) != 301:
                    continue
                line = mp.array([float(line) for num in line[1:]])
                if target_vec is None:
                    target_vec = line
                else:
                    target_vec = np.concatenate((target_vec, line), axis=0)
    print target_vec.shape
    # assert len(vocab) == target_vec.shape[0]

    proj_model = LinearRegression(n_jobs=4)
    proj_model.fit(source_vec, target_vec)
    with open(save_model_file, 'w') as f:
        pickle.dump(proj_model, f)

# TEST ONLY!
def test(test_word):
    with open(vocab_file) as f:
        vocab = f.readlines()
        vocab = [w.strip().lower() for w in vocab]
    print len(vocab)
    vocab2idx = dict([(w,idx) for idx,w in enumerate(vocab)])
    test_idx = vocab2idx[test_word]

    source_vec = np.load(source_vec_file)
    print source_vec.shape

    with open('../../models/skip-thought_linear_projection.m') as f:
        proj_model = pickle.load(f)

    # test similarity on a word
    result = proj_model.predict(source_vec)
    print result.shape
    tups = []
    for w,vec in zip(vocab, result):
        tups.append((vocab[test_idx], w, vec_sim(result[test_idx, :], vec)))
    tups = sorted(tups, key=lambda x: x[2], reverse=True)
    for tup in tups[:100]:
        print("%s\t%s\t%f" % tup)

if __name__ == '__main__':
    # train()
    test('basketball')
