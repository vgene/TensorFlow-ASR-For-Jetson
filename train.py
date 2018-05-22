from __future__ import print_function, division, absolute_import
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import variables # use to initialize
from timeit import default_timer as timer
import tqdm
from model import RNNModel
from data_provider import DataProvider

def build_model(num_layers=1, num_units=128, num_feature=None, keep_prob=1.0):
    if not num_feature:
        print("Please determine number of feature")

    model = RNNModel(num_layers=num_layers, num_units=num_units,
                     num_feature=num_feature, keep_prob=keep_prob, is_training=True)
    model.build_graph()
    loss_op = model.get_loss()
    train_op = model.get_train_op()

def train(model, train_set, batch_size=1, num_epoch=1, learning_rate=1e-3):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(graph=model.graph, config=config) as sess:
        sess.run(variables.global_variables_initializer())

        for epoch in range(num_epoch):
            loss_list = []

            total_iter = int(data_size/batch_size)
            pbar = tqdm.tqdm(total=total_iter)
            iteration = 0

            while True:
                try:
                    iteration += 1
                    #start_time = timer()
                    inputs, labels, seq_lens, is_new_epoch =  train_set.next()
                    loss, _ = sess.run([loss_op, train_op], feed_dict={model.inputs : inputs,
                                                                model.labels : labels,
                                                                model.seq_lens : seq_lens,
                                                                model.learning_rate : 0.001})
                    #end_time = timer()
                    #interval = end_time - start_time
                    pbar.set_description("#{}, Loss:{:.2f}".format(iteration, loss))
                    pbar.update(1)
                    loss_list.append(loss)

                    if is_new_epoch:
                        break
                except KeyboardInterrupt:
                    # TODO: save!
                    avg_loss = np.mean(loss_list)
                    print("Epoch %d: average loss is %f" % (epoch, avg_loss))
                    print("Keyboard interrupt detected, gracefully exiting")
                    return

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

if __name__ == "__main__":
    # model parameters
    model = build_model(num_layers=2,
                        num_units=256,
                        num_feature=120, #for fbank feature
                        keep_prob=0.95)

    train_set = DataProvider(mode='train', batch_size=batch_size)
    data_size = train_set.get_data_size()
    print("Data ready! Data size: %d" % data_size)

    train(model, train_set, batch_size=64, num_epoch=20, learning_rate=1e-3)
