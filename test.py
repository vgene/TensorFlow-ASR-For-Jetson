import tensorflow as tf
import numpy as np
from tensorflow.contrib.cudnn_rnn.python.layers import cudnn_rnn
from tensorflow.python.ops import variables

def test():
    inputs = tf.placeholder(tf.float32, shape=[None, None, 10], name='inputs')
    num = tf.placeholder(tf.float32, name='num')

    # shape0 = tf.shape(inputs)[0]
    # shape1 = tf.shape(inputs)[1]

    # mult = tf.multiply(inputs, num, name='multiply')
    # re_mult = tf.reshape(mult, shape=[shape0*shape1, 16], name='re_mult')

    lstm = cudnn_rnn.CudnnLSTM(
                    num_layers = 2,
                    num_units = 16,
                    direction = 'bidirectional',
                    dropout = 0.0,
                    name = 'cudnn_lstm')
    lstm.build([None, None, 10])

    outputs, states = lstm(inputs, training=True)

    with tf.Session() as sess:
        sess.run(variables.global_variables_initializer())
        raw_inputs = range(180)
        raw_inputs = np.asarray(raw_inputs,dtype="float32", order=None).reshape([6,3,10])
        outputs, states = sess.run([outputs, states], feed_dict={inputs:raw_inputs}, options=None, run_metadata=None)
        print(outputs)

test()
