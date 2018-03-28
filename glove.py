"""
Helper functions to load GloVe vectors & integer-encode sequences of words.
"""

import argparse
import json
import numpy as np
import os
import sys


def load_glove(path):
    """Load pre-trained GloVe vectors into dictionary.

    Args:
        path (str): path to .txt file containing pre-trained GloVe vectors.
    Returns:
        dict:       dictionary mapping word to its vector representation.
    """
    print('Loading embeddings from {}...'.format(path))
    embedding_vectors = {}
    with open(path, 'r') as f:
        for line in f:
            line_split = line.strip().split(' ')
            vector = np.array(line_split[1:], dtype=float)
            word = line_split[0]
            embedding_vectors[word] = vector
    return embedding_vectors


def embedding_matrix(vocab, embed_vectors, embed_size):
    """Create embedding matrix for words in vocabulary.

    Args:
        vocab (dict):         dictionary matching each word in vocabulary to unique integer.
        embed_vectors (dict): dictionary mapping each word to its vector representation.
        embed_size (int):     length of each embedding vector.
    Returns:
        np.ndarray:           embedding matrix with shape (vocab size+2, embedding size)
    """
    matrix = np.zeros((len(vocab)+2, embed_size))  # extra rows for padding, EOF
    for word, i in vocab.items():
        vector = embed_vectors[word]
        assert vector.size == embed_size, 'All embedding vectors must have same length.'
        matrix[i] = vector
    return matrix


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--glove', type=str, default='./models/glove.6B.50d.txt', help='Path to GloVe vectors.')
    parser.add_argument('--emb_size', type=int, default=50, help='Length of embedding vectors.')
    parser.add_argument('--vocab_path', type=str, default='./data/vocab.json', help='Path to vocab file.')
    parser.add_argument('--save_path', type=str, default='./data/word_matrix.npy', help='Where to save word matrix.')
    args = parser.parse_args(sys.argv[1:])

    assert os.path.exists(args.glove), 'Unable to find GloVe vectors. Run setup.sh to download.'
    assert os.path.exists(args.vocab_path), 'Unable to find vocab file. Run data.py to generate data.'

    with open(args.vocab_path, 'r') as file:
        vocab_dict = json.load(file)

    glove = load_glove(args.glove)
    word_matrix = embedding_matrix(vocab_dict, glove, args.emb_size)
    np.save(args.save_path, word_matrix)
