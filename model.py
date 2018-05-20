from __future__ import print_function, division, abosolute_import

import tensorflow as tf
from tensorflow.contrib.cudnn_rnn.python.ops import cudnn_rnn_ops

class Encoder(object):
    """docstring for RNNModel"""
    def __init__(self,
            use_cudnn = True,
            num_layers = 2,
            num_units = 256,
            num_feature = 39,
            keep_prob = 1.0,
            using_conv = False,
            ):
        super(RNNModel, self).__init__()
        # Configurations
        self.use_cudnn = use_cudnn
        self.num_layers = num_layers
        self.num_units = num_units
        # self.batch_size = args.batch_size
        self.num_feature = num_feature
        self.keep_prob = keep_prob

        self.using_conv = using_conv

    def build_graph(self):
        self.input = self.build_input()

        # Start building inputs
        # inputs: [batch_size, time_len, input_size]
        with tf.name_scope("Inputs"):
            self.inputs = tf.placeholder(tf.float32,
                               shape=[None, None, self.num_feature],
                               name='inputs')

            self.labels = tf.SparseTensor(tf.placeholder(tf.int32, name='indices'),
                                tf.placeholder(tf.int32, name='values'),
                                tf.placeholder(tf.int32, name='shape'))

            self.seq_lens = tf.placeholder(tf.int32, shape=[None], name='seq_lens')

            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        if self.using_conv:

        # Start building RNN
        with tf.name_scope("RNN"):
            # TODO: add other LSTM categories
            # Only use cudnnLSTM for now
            self.lstm = cudnn_rnn_ops.CudnnLSTM(
                    num_layers = self.num_layers,
                    num_units = self.num_units,
                    direction = 'bidirectional',
                    dropout = 1.0 - self.keep_prob,
                    name = 'cudnn_lstm')

            # outputs: [time_len, batch_size, num_dirs * num_units]
            # states: a tuple of tensor(s) [num_layers * num_dirs, batch_size, num_units]
            outputs, states = self.lstm(inputs, )

        # Start building fully connected layers, with bottlenneck and FC
        with tf.name_scope("Fully_Connected"):
            output_dim = encoder_outputs.shape.as_list()[-1]
            outputs_2d = tf.reshape(
                encoder_outputs, shape=[batch_size * max_time, output_dim])

            if self.bottleneck_dim is not None and self.bottleneck_dim != 0:
                with tf.variable_scope('bottleneck') as scope:
                    outputs_2d = tf.contrib.layers.fully_connected(
                        outputs_2d,
                        num_outputs=self.bottleneck_dim,
                        activation_fn=tf.nn.relu)

                # Dropout for the hidden-output connections
                outputs_2d = tf.nn.dropout(
                    outputs_2d, keep_prob, name='dropout_bottleneck')

            with tf.variable_scope('output') as scope:
                logits_2d = tf.contrib.layers.fully_connected(
                    outputs_2d,
                    num_outputs=self.num_classes,
                    activation_fn=None)

                if self.time_major:
                    # Reshape back to the original shape
                    logits = tf.reshape(
                        logits_2d, shape=[max_time, batch_size, self.num_classes])
                else:
                    # Reshape back to the original shape
                    logits = tf.reshape(
                        logits_2d, shape=[batch_size, max_time, self.num_classes])

                    # Convert to time-major: `[T, B, num_classes]'
                    logits = tf.transpose(logits, [1, 0, 2])

            self.logits = logits

        # Start building ctc loss
        #TODO: Could add weight decay policy here
        with tf.name_scope("CTC_Loss"):
            # TODO: dig into all variables
            ctc_losses = tf.nn.ctc_loss(
                labels,
                logits,
                #tf.cast(inputs_seq_len, tf.int32),
                inputs_seq_len,
                preprocess_collapse_repeated=False,
                ctc_merge_repeated=True,
                ignore_longer_outputs_than_inputs=True,
                time_major=True)
            self.ctc_loss = tf.reduce_mean(ctc_losses, name='ctc_loss_mean')

    def run_infer(self, inputs, initial_state):



