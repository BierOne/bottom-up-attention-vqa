import os, sys

import numpy as np

sys.path.append(os.getcwd())
from utilities.dataset import Dictionary
from utilities import config, utils


def create_dictionary():
    dictionary = Dictionary()
    if config.type == 'cp':
        ques_file_types = ['train', 'test']
    else:
        ques_file_types = ['train', 'val', 'test']
    for type in ques_file_types:
        questions = utils.get_file(type, question=True)
        for q in questions:
            dictionary.tokenize(q['question'], True)
    return dictionary


def create_glove_embedding_init(idx2word, glove_file):
    word2emb = {}
    with open(glove_file, 'r') as f:
        entries = f.readlines()
    emb_dim = len(entries[0].split(' ')) - 1
    print('embedding dim is %d' % emb_dim)
    weights = np.zeros((len(idx2word)+1, emb_dim), dtype=np.float32) # padding

    for entry in entries:
        vals = entry.split(' ')
        word = vals[0]
        vals = list(map(float, vals[1:]))
        word2emb[word] = np.array(vals)

    for idx, word in enumerate(idx2word):
        if word not in word2emb:
            continue
        weights[idx] = word2emb[word]
    return weights, word2emb


def main():
    d = create_dictionary()
    d.dump_to_file(os.path.join(config.cache_root, 'dictionary.json'))
    d = Dictionary.load_from_file(os.path.join(config.cache_root, 'dictionary.json'))

    emb_dim = 300
    glove_file = os.path.join(config.glove_path, 'glove.6B.{}d.txt'.format(emb_dim))
    weights, word2emb = create_glove_embedding_init(d.idx2word, glove_file)
    np.save(os.path.join(config.cache_root, 'glove6b_init_{}d.npy'.format(emb_dim)), weights)


if __name__ == '__main__':
    main()
