"""
PointerNet Class
"""

import numpy as np
import tensorflow as tf

MAX_LENGTH = 45
BATCH_SIZE = 100

CELL = tf.contrib.rnn.LSTMCell
UNITS = 25
LAYERS = 2



class PointerNet(object):
    def __init__(self):
        word_matrix = np.random.random([200,200])
        # input placeholder for word sequence
        self.encode_input = tf.placeholder(tf.int32, [BATCH_SIZE, MAX_LENGTH], name='encoded_input')
        self.seq_lengths = tf.placeholder(tf.int32, [BATCH_SIZE], name='sequence_lengths')

        # encode the input sequence to glove
        self.word_matrix = tf.Variable(tf.constant(word_matrix), trainable=False, name='word_matrix')
        self.embeded_input = tf.nn.embedding_lookup(self.word_matrix, self.encode_input, name='embeded_input')

        # RNN embedding of the input sequence
        cell = CELL(UNITS)
        init_state = (tf.zeros([BATCH_SIZE, UNITS]), tf.zeros([BATCH_SIZE, UNITS]))
        cell_fw = tf.contrib.rnn.MultiRNNCell([cell for _ in range(LAYERS)])

        self.outputs, state = tf.nn.dynamic_rnn(cell_fw, self.embeded_input, sequence_length=self.seq_lengths, initial_state=init_state,
                                                dtype=tf.float32)


if __name__ == '__main__':
    model = PointerNet()
    # print(model.outputs)
