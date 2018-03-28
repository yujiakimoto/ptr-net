"""
Implementation of Pointer Network using AttentionWrapper.
"""

import numpy as np
import tensorflow as tf
from tensorflow.contrib.seq2seq import BahdanauAttention

MAX_LENGTH = 45
BATCH_SIZE = 100
LEARNING_RATE = 0.0001

ENCODER_CELL = tf.contrib.rnn.LSTMCell
ENCODER_UNITS = 25
ENCODER_LAYERS = 2

ATTENTION = BahdanauAttention
ATTN_SIZE = 30
N_POINTERS = 1

DECODER_CELL = tf.contrib.rnn.LSTMCell
DECODER_UNITS = 50
DECODER_LAYERS = 3


class PointerNet(object):
    def __init__(self):
        with tf.variable_scope('inputs'):
            # load pre-trained GloVe embeddings
            word_matrix = tf.constant(np.load('./data/word_matrix.npy'), dtype=tf.float32)
            self.word_matrix = tf.Variable(word_matrix, trainable=False, name='word_matrix')

            # input placeholder for sequence of words
            self.inputs = tf.placeholder(tf.int32, [BATCH_SIZE, MAX_LENGTH], name='inputs')
            # actual (non-padded) length of each sequence in batch; used in dynamic_rnn()
            self.input_lengths = tf.placeholder(tf.int32, [BATCH_SIZE], name='input_lengths')

        with tf.variable_scope('outputs'):
            # output placeholder for labels
            self.outputs = tf.placeholder(tf.int32, [BATCH_SIZE, N_POINTERS], name='outputs')
            self.labels = tf.unstack(self.outputs, axis=1)
            # sequence length for pointers is always equal to the number of pointers
            self.output_lengths = tf.constant(np.full(BATCH_SIZE, N_POINTERS), dtype=tf.int32)

        with tf.variable_scope('embeddings'):
            # encode each word in sequence as 50-dimensional vector
            self.embedded = tf.nn.embedding_lookup(self.word_matrix, self.inputs, name='embedded')

        with tf.variable_scope('encoder'):
            # RNN embedding of the input sequence
            enc_cell = tf.contrib.rnn.MultiRNNCell([ENCODER_CELL(ENCODER_UNITS) for _ in range(ENCODER_LAYERS)])
            init_state = enc_cell.zero_state(BATCH_SIZE, dtype=tf.float32)
            self.encoded, enc_state = tf.nn.dynamic_rnn(enc_cell, self.embedded, sequence_length=self.input_lengths,
                                                        initial_state=init_state)

        with tf.variable_scope('decoder'):
            attention = ATTENTION(N_POINTERS, self.encoded, memory_sequence_length=self.input_lengths)
            dec_cell = tf.contrib.rnn.MultiRNNCell([DECODER_CELL(DECODER_UNITS) for _ in range(DECODER_LAYERS)])
            attn_cell = tf.contrib.seq2seq.AttentionWrapper(dec_cell, attention, attention_layer_size=ATTN_SIZE,
                                                            alignment_history=True)
            init_state = attn_cell.zero_state(BATCH_SIZE, dtype=tf.float32)
            helper = tf.contrib.seq2seq.TrainingHelper(tf.cast(tf.expand_dims(self.outputs, 2), tf.float32),
                                                       self.output_lengths)
            decoder = tf.contrib.seq2seq.BasicDecoder(attn_cell, helper, init_state)
            _, states, _ = tf.contrib.seq2seq.dynamic_decode(decoder, maximum_iterations=N_POINTERS)

        with tf.variable_scope('pointers'):
            # tensor of shape (# pointers, batch size, max. input sequence length)
            self.pointer_prob = tf.reshape(states.alignment_history.stack(), [N_POINTERS, BATCH_SIZE, MAX_LENGTH])
            self.pointers = tf.unstack(tf.argmax(self.pointer_prob, axis=2, output_type=tf.int32))

        with tf.variable_scope('loss'):
            self.loss = tf.zeros(())
            pointers = tf.unstack(self.pointer_prob)

            equal = []
            for i in range(N_POINTERS):
                self.loss += tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels[i], logits=pointers[i])
                equal.append(tf.equal(self.pointers[i], self.labels[i]))
            self.correct = tf.cast(tf.stack(equal), tf.float32)
            self.all_correct = tf.cast(tf.equal(tf.reduce_sum(self.correct, axis=0), N_POINTERS), tf.float32)
            self.exact_match = tf.reduce_mean(self.all_correct)

        with tf.variable_scope('training'):
            self.train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(self.loss)


if __name__ == '__main__':
    model = PointerNet()

