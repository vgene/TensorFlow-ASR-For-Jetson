from __future__ import print_function, division, absolute_import
import os
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import variables # use to initialize
from model import RNNModel
from data_provider import DataProvider
import warpctc_tensorflow
import tqdm

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'

num_epoch = 20

num_layers = 2
num_units = 256
num_feature = 120

model = RNNModel(num_layers=num_layers, num_units=num_units, num_feature=num_feature, keep_prob=0.95, is_training=True)
model.build_graph()
loss_op = model.get_loss()
train_op = model.get_train_op()

with tf.Session(target='', graph=None, config=None) as sess:
    sess.run(variables.global_variables_initializer())

    for epoch in range(num_epoch):
        loss_list = []

        train_set = DataProvider('train') # under implementation

        while:
            inputs, labels, seq_lens, is_new_epoch =  train_set.next()
            loss, _ = sess.run([loss_op, train_op], feed_dict={model.inputs : inputs,
                                                        model.labels : labels,
                                                        model.seq_lens : seq_lens,
                                                        model.learning_rate : 0.001})
            if is_new_epoch:
                break
        loss_list.append(loss)
        print(loss)


        '''
        for inputs, labels, seq_lens in train_set.next():
            loss, _ = sess.run([loss, train_op], feed_dict={model.inputs = inputs,
                                                              model.labels = labels,
                                                              model.seq_lens = seq_lens
                                                              model.learning_rate = 1e-3})
            loss_list.append(loss)
            print(loss)
        '''

        avg_loss = np.mean(loss_list)

        print("Epoch %d: average loss is %f" % (epoch, avg_loss))

