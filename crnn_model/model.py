"""
Implement for the crnn model mentioned in "An End-to-End Trainable Neural Network for Image-based Sequence
Recognition and Its Application to Scene Text Recognition"

https://arxiv.org/abs/1507.05717v1
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib import slim

_BATCH_DECAY = 0.999

class CRNNCTCNetwork(object):
    def __init__(self, phase, hidden_num, layers_num, num_classes):
        self.__phase = phase.lower()
        self.__hidden_num = hidden_num
        self.__layers_num = layers_num
        self.__num_classes = num_classes
        return

    def __feature_sequence_extraction(self, input_tensor):
        is_training = True if self.__phase == 'train' else False
        with slim.arg_scope([slim.conv2d], padding='SAME',
                            weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                            weights_regularizer=slim.l2_regularizer(0.0005),
                            biases_initializer=None):
            net = slim.repeat(input_tensor, 2, slim.conv2d, 64, kernel_size=3, stride=1, scope='conv1') # [batch,32, 100, 64]
            net = slim.max_pool2d(net, kernel_size=2, stride=2, scope='pool1') # [batch, 16, 50, 64]
            net = slim.repeat(net, 2, slim.conv2d, 128, kernel_size=3, stride=1, scope='conv2') # [batch, 16, 50, 128]
            net = slim.max_pool2d(net, kernel_size=2, stride=2, scope='pool2') # [batch, 8, 25, 128]
            net = slim.repeat(net, 2, slim.conv2d, 256, kernel_size=3, stride=1, scope='conv3') # [batch, 8, 25, 256]
            net = slim.max_pool2d(net, kernel_size=[2, 1], stride=[2, 1], scope='pool3') # [batch, 4, 25, 256]
            net = slim.conv2d(net, 512, kernel_size=3, stride=1, scope='conv4') # [batch, 4, 25, 256]
            net = slim.batch_norm(net, decay=_BATCH_DECAY, is_training=is_training, scope='bn4') # [batch, 4, 25, 256]
            net = slim.conv2d(net, 512, kernel_size=3, stride=1, scope='conv5') # [batch, 4, 25, 512]
            net = slim.batch_norm(net, decay=_BATCH_DECAY, is_training=is_training, scope='bn5') # batch*4*25*512
            net = slim.max_pool2d(net, kernel_size=[2, 1], stride=[2, 1], scope='pool5') # [batch, 2, 25, 512]
            net = slim.conv2d(net, 512, kernel_size=2, stride=[2, 1], scope='conv6') # [batch, 1, 25, 512]
        return net
    
    def __map_to_sequence(self, input_tensor):
        shape = input_tensor.get_shape().as_list()
        assert shape[1] == 1  # H of the feature map must equal to 1
        return tf.squeeze(input_tensor, axis=1)

    def __sequence_label(self, input_tensor):
        with tf.variable_scope('LSTM_Layers'):
            # forward lstm cell
            fw_cell_list = [rnn.BasicLSTMCell(nh, forget_bias=1.0) for nh in [self.__hidden_num]*self.__layers_num]
            # Backward direction cells
            bw_cell_list = [rnn.BasicLSTMCell(nh, forget_bias=1.0) for nh in [self.__hidden_num]*self.__layers_num]
            stack_lstm_layer, _, _ = rnn.stack_bidirectional_dynamic_rnn(fw_cell_list, bw_cell_list, input_tensor, dtype=tf.float32)

            [batch_size, _, hidden_num] = input_tensor.get_shape().as_list()  # [batch, width, 2*hidden_num] = [batch, 25, 512]
            rnn_reshaped = tf.reshape(stack_lstm_layer, [-1, hidden_num])  # [batch x width, 2*hidden_num]

            # Doing the affine projection
            w = tf.Variable(tf.truncated_normal([hidden_num, self.__num_classes], stddev=0.01), name="w")
            logits = tf.matmul(rnn_reshaped, w)
            
            logits = tf.reshape(logits, [batch_size, -1, self.__num_classes])
            raw_pred = tf.argmax(tf.nn.softmax(logits), axis=2, name='raw_prediction')

            # Swap batch and batch axis
            rnn_out = tf.transpose(logits, (1, 0, 2), name='transpose_time_major')  # [width, batch, n_classes] 
        return rnn_out, raw_pred

    def build_network(self, input_tensor):
        # first apply the cnn feature extraction stage
        cnn_out = self.__feature_sequence_extraction(input_tensor=input_tensor)
        # second apply the map to sequence stage
        sequence = self.__map_to_sequence(input_tensor=cnn_out)
        # third apply the sequence label stage
        net_out, raw_pred = self.__sequence_label(input_tensor=sequence)
        return net_out
