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
python data.py   # generate dataset
python glove.py  # generate embedding matrix of only words in vocab
python train.py  # train model
```

## Performance

Our model achieves 100% test accuracy on this synthetic task.