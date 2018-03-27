"""
Generate synthetic data.
"""

import argparse
import json
import numpy as np
import os
import random
import sys

prep = ['', 'to', 'to the']
dest = {'': ['home', 'nowhere'],
        'to': ['japan', 'school', 'germany', 'canada'],
        'to the': ['stadium', 'station', 'hospital', 'classroom', 'kitchen', 'morgue']}
advb = ['slowly', 'quickly', 'suddenly', 'reluctantly', 'happily']
when = ['ten minutes ago', 'five minutes ago', 'yesterday', 'earlier']
subj = ['she', 'he']


def tokenize(text):
    """Tokenize a passage of text, i.e. return a list of words"""
    text = text.replace('.', '')
    return text.split(' ')


def generate_sentence(s=None):
    """Generate a random sentence."""
    # if subject is not specified, sample randomly from set
    s = s or random.sample(subj, 1)[0]
    v = 'went'
    p = random.sample(prep, 1)[0]
    d = random.sample(dest[p], 1)[0]
    a = random.sample(advb, 1)[0]
    w = random.sample(when, 1)[0]

    # insert adverb before verb with prob. 40%, after destination with prob. 40%, no adverb with prob. 20%
    r1 = np.random.rand()
    if r1 < 0.2:
        sentence = ' '.join([s, v, p, d])     # e.g. 'she went to school'
    elif r1 < 0.6:
        sentence = ' '.join([s, a, v, p, d])  # e.g. 'she reluctantly went to school'
    else:
        sentence = ' '.join([s, v, p, d, a])  # e.g. 'she went to school reluctantly'

    # insert clause specifying time with probability 50%
    r2 = np.random.rand()
    if r2 < 0.5:
        sentence += ' ' + w                   # e.g. 'she went to school reluctantly yesterday'

    sentence = sentence.replace('  ', ' ')
    # record index of destination only if subject is 'she'
    if s == 'she':
        index = sentence.split(' ').index(d)
    else:
        index = 0
    return sentence + '.', index


def generate_passage(n):
    """Generate a passage composed of n random sentences."""
    # ensures that at least one of n sentences has the subject 'she'
    data = [generate_sentence('she')]
    for _ in range(n-1):
        data.append(generate_sentence())
    random.shuffle(data)

    sentences = ' '.join(d[0] for d in data)   # join all sentences to create passage
    indices = [d[1] for d in data]             # indices of destination of 'she' (index is relative to each sentence)

    # sentence containing destination is final sentence for which 'she' is the subject
    target_sentence = max([i for i, e in enumerate(indices) if e > 0])
    index = 0
    for i in range(target_sentence):
        index += len(tokenize(data[i][0]))     # add number of words in all sentences preceding target sentence
    index += indices[target_sentence]          # add of target word in target sentence
    return sentences, index


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_passages', type=int, default=5000, help='Number of passages to generate.')
    parser.add_argument('--data_path', type=str, default='./data/train.txt', help='Where to save training data.')
    parser.add_argument('--vocab_path', type=str, default='./data/vocab.json', help='Where to save vocabulary.')
    args = parser.parse_args(sys.argv[1:])

    # if file at args.data_path exists already, delete it
    try:
        os.remove(args.data_path)
    except OSError:
        pass

    # write training data to txt file; save vocabulary in json file
    vocab = set()
    with open(args.data_path, 'a') as file:
        for _ in range(args.n_passages):
            n_sentences = np.random.randint(1, 5)
            passage, idx = generate_passage(n_sentences)
            file.write('{} {}\n'.format(idx, passage))
            vocab = vocab.union(set(tokenize(passage)))

    with open(args.vocab_path, 'w') as file:
        vocab_dict = {word: i+1 for i, word in enumerate(list(vocab))}  # reserve 0 for padding
        json.dump(vocab_dict, file)
