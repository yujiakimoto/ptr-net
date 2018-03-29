"""
Generate synthetic data.
"""

import argparse
import json
import numpy as np
import os
import random
import re
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


class Loader(object):
    """Text data loader."""
    def __init__(self, data_path, vocab_path, batch_size, seq_length):
        self.data_path = data_path
        self.batch_size = batch_size
        self.seq_length = seq_length

        with open(vocab_path, 'r') as f:
            self.vocab = json.load(f)
            self.vocab_size = len(self.vocab)

        self.embedding = None
        self.lengths = None
        self.labels = None
        self.n_batches = None
        self.x_batches, self.x_lengths, self.y_batches = None, None, None
        self.pointer = 0

        print('Pre-processing data...')
        self.pre_process()
        self.create_batches()
        print('Pre-processed {} lines of data.'.format(self.labels.shape[0]))

    def pre_process(self):
        """Pre-process data."""
        with open(self.data_path, 'r') as f:
            data = f.readlines()
        # each line in data file is formatted according to [label, text] (e.g. 2 She went home)
        text = [sample[2:].strip() for sample in data]
        self.labels = np.array([[int(n) for n in re.findall(r'\d+', sample)] for sample in data])
        self.embedding = np.zeros((len(text), self.seq_length), dtype=int)
        self.lengths = np.zeros(len(text), dtype=int)
        for i, sample in enumerate(text):
            tokens = tokenize(text[i])
            self.lengths[i] = len(tokens)
            self.embedding[i] = list(map(self.vocab.get, tokens)) + [1] + [0] * (self.seq_length - len(tokens) - 1)

    def create_batches(self):
        """Split data into training batches."""
        self.n_batches = int(self.embedding.shape[0] / self.batch_size)
        # truncate training data so it is equally divisible into batches
        self.embedding = self.embedding[:self.n_batches * self.batch_size, :]
        self.lengths = self.lengths[:self.n_batches * self.batch_size]
        self.labels = self.labels[:self.n_batches * self.batch_size, :]

        # split training data into equal sized batches
        self.x_batches = np.split(self.embedding, self.n_batches, 0)
        self.x_lengths = np.split(self.lengths, self.n_batches)
        self.y_batches = np.split(self.labels, self.n_batches, 0)

    def next_batch(self):
        """Return current batch, increment pointer by 1 (modulo n_batches)"""
        x, x_len, y = self.x_batches[self.pointer], self.lengths[self.pointer], self.y_batches[self.pointer]
        self.pointer = (self.pointer + 1) % self.n_batches
        return x, x_len, y


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_passages', type=int, default=5000, help='Number of passages to generate.')
    parser.add_argument('--data_path', type=str, default='./data', help='Where to save training data.')
    parser.add_argument('--vocab_path', type=str, default='./data/vocab.json', help='Where to save vocabulary.')
    args = parser.parse_args(sys.argv[1:])

    # if files at args.data_path exist already, delete it
    for file in ['train.txt', 'validate.txt', 'test.txt']:
        path = os.path.join(args.data_path, file)
        if os.path.exists(path):
            os.remove(path)

    # write training, validate, test data to txt files; save vocabulary in json file
    vocab = set()
    with open(os.path.join(args.data_path, 'train.txt'), 'a') as file:
        for _ in range(args.n_passages):
            n_sentences = np.random.randint(1, 5)
            passage, idx = generate_passage(n_sentences)
            file.write('{} {}\n'.format(idx, passage))
            vocab = vocab.union(set(tokenize(passage)))

    with open(os.path.join(args.data_path, 'validate.txt'), 'a') as file:
        for _ in range(int(args.n_passages/5)):
            n_sentences = np.random.randint(1, 5)
            passage, idx = generate_passage(n_sentences)
            file.write('{} {}\n'.format(idx, passage))
            vocab = vocab.union(set(tokenize(passage)))

    with open(os.path.join(args.data_path, 'test.txt'), 'a') as file:
        for _ in range(int(args.n_passages/5)):
            n_sentences = np.random.randint(1, 5)
            passage, idx = generate_passage(n_sentences)
            file.write('{} {}\n'.format(idx, passage))
            vocab = vocab.union(set(tokenize(passage)))

    with open(args.vocab_path, 'w') as file:
        vocab_dict = {word: i+2 for i, word in enumerate(list(vocab))}  # reserve 0 for padding, 1 for EOF
        json.dump(vocab_dict, file)
