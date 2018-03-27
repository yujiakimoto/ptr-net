"""
Generate synthetic data.
"""

import numpy as np
import random

prep = ['', 'to', 'to the']
dest = {'': ['home', 'nowhere'],
        'to': ['Japan', 'school', 'Germany', 'Canada'],
        'to the': ['stadium', 'station', 'hospital', 'classroom', 'kitchen', 'morgue']}
advb = ['slowly', 'quickly', 'suddenly', 'reluctantly', 'happily']
when = ['ten minutes ago', 'five minutes ago', 'yesterday', 'earlier']
subj = ['She', 'He']


def generate_sentence(s=None):
    """Generate a random sentence."""
    # if subject is not specified, sample randomly from set
    s = s or random.sample(subj, 1)[0]
    v = 'went'
    p = random.sample(prep, 1)[0]
    d = random.sample(dest[p], 1)[0]
    a = random.sample(advb, 1)[0]
    w = random.sample(when, 1)[0]

    # insert adverb before verb with prob. 40%, after destination with prob. 40%, no adverb with prob 20%
    r1 = np.random.rand()
    if r1 < 0.2:
        sentence = ' '.join([s, v, p, d])
    elif r1 < 0.6:
        sentence = ' '.join([s, a, v, p, d])
    else:
        sentence = ' '.join([s, v, p, d, a])

    # insert clause specifying time with probability 50%
    r2 = np.random.rand()
    if r2 < 0.5:
        sentence += ' ' + w

    sentence = sentence.replace('  ', ' ')
    # record index of destination only if subject is 'She'
    if s == 'She':
        index = sentence.split(' ').index(d)
    else:
        index = 0

    return sentence, index


def passage(n):
    """Generate n random sentences."""
    s = [generate_sentence('She')]
    for _ in range(n-1):
        s.append(generate_sentence())
    random.shuffle(s)
    sentences = ' '.join(t[0] for t in s)

    indices = [t[1] for t in s]
    nonzero = [i for i, e in enumerate(indices) if e > 0]

    # final destination of 'she' is the destination at final sentence of which she is the subject
    index = 0
    for i in range(max(nonzero)):
        index += len(s[i][0].split(' '))
    index += s[max(nonzero)][1]
    return sentences, index
