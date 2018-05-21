import tensorflow as tf
import numpy as np
from tensorflow.contrib.cudnn_rnn.python.ops import cudnn_rnn_ops

def test():
    inputs = tf.placeholder(tf.float32, shape=[None, None, 10], name='inputs')
    num = tf.placeholder(tf.float32, name='num')

    # shape0 = tf.shape(inputs)[0]
    # shape1 = tf.shape(inputs)[1]

    # mult = tf.multiply(inputs, num, name='multiply')
    # re_mult = tf.reshape(mult, shape=[shape0*shape1, 16], name='re_mult')

    lstm = cudnn_rnn_ops.CudnnLSTM(
                    num_layers = 2,
                    num_units = 16,
                    direction = 'bidirectional',
                    dropout = 0.0,
                    name = 'cudnn_lstm')

    outputs, states = lstm(inputs, training=self.is_training)

    with tf.Session() as sess:

        raw_inputs = range(180)
        raw_inputs = np.asarray(raw_inputs,dtype="float32", order=None).reshape([6,3,10])
        outputs, states = sess.run([outputs, states], feed_dict={'inputs':inputs}, options=None, run_metadata=None)
        print(outputs)