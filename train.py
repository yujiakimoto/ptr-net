"""
Training Script
"""
import model
from model import PointerNet
from glove import encode
import json
from sklearn.model_selection import train_test_split
from data import tokenize
import tensorflow as tf
import numpy as np
from tqdm import tqdm

def train():
    # TODO : parse user argument
    # load train file
    with open('data/train.txt', 'r') as f:
        data = f.readlines()
    # load dictionary
    with open('data/vocab.json', 'r') as f:
        word_dict = json.load(f)

    # process data
    contents = [x[2:].strip() for x in data]

    # dataset = encode(word_dict,contents,model.MAX_LENGTH)
    labels = [int(x[:1].strip()) for x in data]

    # Train / validation / test split
    X_train, X_test, y_train, y_test = train_test_split(contents, labels, test_size = 0.2)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.1)

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_val = np.array(X_val)
    y_val = np.array(y_val)
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    # Training iteration
    ptr_model = PointerNet()
    saver = tf.train.Saver()
    # number of iteration per epoch
    total_batch = int(len(X_train) / model.BATCH_SIZE)
    val_batch = int(len(X_val) / model.BATCH_SIZE)
    # best validation EM
    best_val_acc = 0.0
    # train loss
    train_loss = tf.reduce_mean(ptr_model.loss)
    # initialize graph
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for i in tqdm(range(model.EPOCH)):
            # shuffle data
            idx = np.random.permutation(len(X_train))
            X_train = X_train[idx]
            y_train = y_train[idx]
            inputs = encode(word_dict, X_train, model.MAX_LENGTH)
            input_lengths = [len(tokenize(x)) for x in X_train]
            # train iteration
            for itr in range(total_batch):
                x_in = inputs[itr*model.BATCH_SIZE:itr*model.BATCH_SIZE+model.BATCH_SIZE]
                in_len = input_lengths[itr*model.BATCH_SIZE:itr*model.BATCH_SIZE+model.BATCH_SIZE]
                out = y_train[itr*model.BATCH_SIZE:itr*model.BATCH_SIZE+model.BATCH_SIZE]

                l,em,_ = sess.run([train_loss, ptr_model.exact_match, ptr_model.train_step], feed_dict={
                    ptr_model.inputs: x_in,
                    ptr_model.input_lengths: in_len,
                    ptr_model.outputs : out.reshape(-1,1)})

            if i % 100 == 0:
                idx = np.random.permutation(len(X_val))
                X_val = X_val[idx]
                y_val = y_val[idx]
                val_inputs = encode(word_dict, X_val, model.MAX_LENGTH)
                val_input_lengths = [len(tokenize(x)) for x in X_val]

                for itr in range(val_batch):
                    x_in = val_inputs[itr*model.BATCH_SIZE:itr*model.BATCH_SIZE+model.BATCH_SIZE]
                    in_len = val_input_lengths[itr*model.BATCH_SIZE:itr*model.BATCH_SIZE+model.BATCH_SIZE]
                    out = y_val[itr*model.BATCH_SIZE:itr*model.BATCH_SIZE+model.BATCH_SIZE]

                    v_em = sess.run(ptr_model.exact_match, feed_dict={
                        ptr_model.inputs: x_in,
                        ptr_model.input_lengths: in_len,
                        ptr_model.outputs : out.reshape(-1,1)})


                print("{} epoch, loss {:.2f}".format(i, l))
                print("Train EM : {:.2f}, Validation EM : {:.2f}".format(em, v_em))

                # save model
                if v_em > best_val_acc:
                    save_path = saver.save(sess, model.SAVE_DIR)
                    best_val_acc = v_em
                    print("Model saved")

        print("Training Done")
        print("Best Validation EM : {:.2f}".format(best_val_acc))




if __name__ == "__main__":
    train()
