from __future__ import print_function, division, absolute_import

import tensorflow as tf
from tensorflow.contrib.cudnn_rnn.python.layers import cudnn_rnn
import warpctc_tensorflow
#from tensorflow.contrib.rnn.python.layers import

class RNNModel(object):
    """RNNModel"""
    def __init__(self,
                 use_cudnn = True,
                 num_layers = 2,
                 num_units = 256,
                 num_feature = 39,
                 keep_prob = 1.0,
                 using_conv = False,
                 is_training = True,
                 time_major = True,
                 num_classes = 28
            ):
        super(RNNModel, self).__init__()
        # Configurations
        self.use_cudnn = use_cudnn # Using cudnnLSTM or not
        self.num_layers = num_layers # Number of layers in LSTM
        self.num_units = num_units # Number of units in one layer of LSTM
        # self.batch_size = args.batch_size
        self.num_feature = num_feature # Feature input of LSTM
        self.keep_prob = keep_prob # Dropout keep probability
        self.using_conv = using_conv # Whether use convolution layer to preprocess feature
        self.is_training = is_training # Determine training or inference
        self.time_major = time_major # Determine whether using time as first dimension
        self.num_classes = num_classes+1 # Add one blank (TODO: as 0 or last one?)

    def build_graph(self):
        # Start building inputs
        # inputs: [time_len, batch_size, input_size]
        with tf.name_scope("Inputs"):
            self.inputs = tf.placeholder(tf.float32,
                               shape=[None, None, self.num_feature],
                               name='inputs')

            self.labels = tf.SparseTensor(tf.placeholder(tf.int64, name='indices'),
                                tf.placeholder(tf.int32, name='values'),
                                tf.placeholder(tf.int64, name='shape'))

            self.seq_lens = tf.placeholder(tf.int32, shape=[None], name='seq_lens')

            self.learning_rate = tf.placeholder(tf.float32, name='learning_rate')

            # use __init__ variables instead
            # self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        # Convolution Preprocessing
        # if self.using_conv:
        #     with tf.name_scope("Convolution"):
        #         self.conv = tf.nn.conv2d()

        # Start building RNN
        with tf.name_scope("RNN"):
            # TODO: add other LSTM categories
            # Only use cudnnLSTM for now
            if self.use_cudnn:
                self.lstm = cudnn_rnn.CudnnLSTM(
                        num_layers = self.num_layers,
                        num_units = self.num_units,
                        direction = 'bidirectional',
                        dropout = 1.0 - self.keep_prob,
                        name = 'cudnn_lstm')

                # build first(optional)
                self.lstm.build([None, None, self.num_feature])
                self.outputs, self.states = self.lstm(self.inputs, training=self.is_training)
            # else:
            #     # CudnnCompatibleLSTMCell
            #     self.lstm = cud

            # input: [time_len, batch_size, input_size]
            # outputs: [time_len, batch_size, num_dirs * num_units]
            # states: a tuple of tensor(s) [num_layers * num_dirs, batch_size, num_units]

            self.encoder_outputs = self.outputs

        # Start building fully connected layers, with bottlenneck and FC
        with tf.name_scope("Fully_Connected"):
            batch_size = tf.shape(self.inputs)[1]
            max_time = tf.shape(self.inputs)[0]
            output_dim = self.encoder_outputs.shape.as_list()[-1]

            outputs_2d = tf.reshape(
                self.encoder_outputs, shape=[batch_size * max_time, output_dim])

            # if self.bottleneck_dim is not None and self.bottleneck_dim != 0:
            #     with tf.variable_scope('bottleneck') as scope:
            #         outputs_2d = tf.contrib.layers.fully_connected(
            #             outputs_2d,
            #             num_outputs=self.bottleneck_dim,
            #             activation_fn=tf.nn.relu)

            #     # Dropout for the hidden-output connections
            #     outputs_2d = tf.nn.dropout(
            #         outputs_2d, keep_prob, name='dropout_bottleneck')

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
            #self.logits =tf.Print(self.logits, [tf.shape(self.logits)])

        # Start building ctc loss
        # TODO: Could add weight decay policy here
        with tf.name_scope("CTC_Loss"):
            # TODO: dig into all variables
            # labels: int32 SparseTensor.
            #         labels.indices[i, :] == [b, t] means labels.values[i] stores the id for (batch b, time t).
            #         labels.values[i] must take on values in [0, num_labels)
            # logits: 3-D float Tensor [max_time, batch_size, num_classes]
            # inputs_seq_len: 1-D int32 vector, [batch_size]

            # return 1-D float tensor: [batch], neg-log prob
            ctc_losses = tf.nn.ctc_loss(
                self.labels,
                self.logits,
                #tf.cast(inputs_seq_len, tf.int32),
                self.seq_lens,
                preprocess_collapse_repeated=False,
                ctc_merge_repeated=True,
                ignore_longer_outputs_than_inputs=True,
                time_major=True)
            self.ctc_loss = tf.reduce_mean(ctc_losses, name='ctc_loss_mean')

        # TODO: add more optimizers
        with tf.name_scope("Optimizer"):
            self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate, decay=0.9, momentum=0.0, epsilon=1e-10, use_locking=False, centered=False)

        if self.is_training:
            self.train_op = self.optimizer.minimize(self.ctc_loss)

    def get_train_op(self):
        # Please build_graph first!
        return self.train_op

    def get_loss(self):
        # Please build_graph first!
        return self.ctc_loss

    def get_logits(self):
        # Please build_graph first!
        return self.logits

    def decode(self, logits, inputs_seq_len, beam_width=1):
        """Operation for decoding.
        Args:
            logits: A tensor of size `[T, B, num_classes]`
            inputs_seq_len: A tensor of size `[B]`
            beam_width (int, optional): beam width for beam search.
                1 disables beam search, which mean greedy decoding.
        Return:
            decode_op: A SparseTensor
        """
        assert isinstance(beam_width, int), "beam_width must be integer."
        assert beam_width >= 1, "beam_width must be >= 1"

        # inputs_seq_len = tf.cast(inputs_seq_len, tf.int32)

        if beam_width == 1:
            decoded, neg_sum_logits = tf.nn.ctc_greedy_decoder(
                logits, inputs_seq_len)
        else:
            decoded, neg_sum_logits = tf.nn.ctc_beam_search_decoder(
                logits, inputs_seq_len,
                beam_width=beam_width)

        decode_op = tf.to_int32(decoded[0])

        return decode_op, neg_sum_logits



