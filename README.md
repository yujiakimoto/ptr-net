# Pointer Networks
TensorFlow implementation of a Pointer Network (https://arxiv.org/abs/1506.03134)

## Introduction
Pointer networks are a neural network architecture that deal with
problems whose output dimensions depend on the length of the input.
They have found success in QA systems, where the answer to a question
can be identified by two pointers (start & end) in a passage of text.
   
We have created a synthetic dataset for which a pointer net is suitable:
a sequence of random sentences involving two subjects "he" and "she", where
the label is the final location of "she".  

```
he went home happily ten minutes ago. she went home slowly yesterday.
                                               ^^^^
she went to germany yesterday. she went home quickly.
                                        ^^^^
she happily went to japan earlier. he suddenly went to the kitchen yesterday.
                    ^^^^^                                     
``` 

## Installation & Training
We use [GloVe](https://nlp.stanford.edu/projects/glove/) to generate
vector representations of words. To install pre-trained vectors, run
```
./setup.sh
```
The following is optional (to be run if you want to generate custom data, 
re-train the model etc.)
```
python data.py     # generate dataset
python glove.py    # generate embedding matrix of only words in vocab
python train.py    # train model
```

## TensorFlow Model

Pointer networks are a simplification of seq2seq models with attention.
An encoder network reads the input data, producing an embedding vector
for each input. A decoder network then begins generating outputs - with 
a pointer network, the softmaxed attention weights over the inputs are used
as probabilistic "pointers" that point to an element in the input sequence.  
[ptr-net](images/ptr-net.png) 


The inputs to the pointer network are integer-encoded passages. To account
for the fact that each input passage may be of variable length, we also
input the actual, non-padded length of each sequence. This allows for dynamic
unrolling of the encoder network.
```python 
self.encoder_inputs = tf.placeholder(tf.int32, [batch_size, seq_length])
self.input_lengths = tf.placeholder(tf.int32, [batch_size])
```

At training time, we supply the labelled pointers (in our case, the index of
the word that is the answer). The inputs to the decoder network are a special
start token, followed by the sequence of inputs that the true pointers point to.
This is achieved using a combination of `tf.stack`/`tf.unstack` and `tf.gather`.
The expected outputs of the decoder network are its inputs, shifted over one 
time step.
```python
self.pointer_labels = tf.placeholder(tf.int32, [batch_size, n_pointers])
start_tokens = tf.constant(START_TOKEN, shape=[batch_size], dtype=tf.int32)
self.decoder_labels = tf.stack([tf.gather(inp, ptr) for inp, ptr in list(zip(tf.unstack(self.encoder_inputs), tf.unstack(self.pointer_labels)))])
self.decoder_inputs = tf.concat([tf.expand_dims(start_tokens, 1), self.decoder_labels], 1)
self.output_lengths = tf.constant(n_pointers, shape=[batch_size])
```

```python
word_matrix = tf.constant(np.load('./data/word_matrix.npy'), dtype=tf.float32)
self.word_matrix = tf.Variable(word_matrix, trainable=True, name='word_matrix')
self.input_embeds = tf.nn.embedding_lookup(self.word_matrix, self.encoder_inputs)
self.output_embeds = tf.nn.embedding_lookup(self.word_matrix, self.decoder_inputs)
```

## Performance

Our model achieves 100% test accuracy on this synthetic task.