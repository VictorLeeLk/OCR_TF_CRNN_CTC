from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import json

import tensorflow as tf

import cv2
import numpy as np

from crnn_model import model

os.environ["CUDA_VISIBLE_DEVICES"]=""

_IMAGE_SIZE = (100, 32)
_SEQUEENCE_LENGTH = 25 # 100 /4

# ------------------------------------Basic prameters------------------------------------
tf.app.flags.DEFINE_string(
    'image_dir', './test_data/images/', 'Path to the directory containing images.')

tf.app.flags.DEFINE_string(
    'image_list', './test_data/image_list.txt', 'Path to the images list txt file.')

tf.app.flags.DEFINE_string(
    'model_dir', './model/', 'Base directory for the model.')

# ------------------------------------LSTM prameters------------------------------------
tf.app.flags.DEFINE_integer(
    'lstm_hidden_layers', 2, 'The number of stacked LSTM cell.')

tf.app.flags.DEFINE_integer(
    'lstm_hidden_uints', 256, 'The number of units in each LSTM cell')

# ------------------------------------Char dictionary------------------------------------

tf.app.flags.DEFINE_string(
    'char_map_json_file', './char_map/char_map.json', 'Path to char map json file')

FLAGS = tf.app.flags.FLAGS

def _sparse_matrix_to_list(sparse_matrix, char_map_dict=None):
    indices = sparse_matrix.indices
    values = sparse_matrix.values
    dense_shape = sparse_matrix.dense_shape

    # the last index in sparse_matrix is ctc blanck note
    if char_map_dict is None:
        char_map_dict = json.load(open(FLAGS.char_map_json_file, 'r'))
    assert(isinstance(char_map_dict, dict) and 'char_map_dict is not a dict')    

    dense_matrix =  len(char_map_dict.keys()) * np.ones(dense_shape, dtype=np.int32)
    for i, indice in enumerate(indices):
        dense_matrix[indice[0], indice[1]] = values[i]
    string_list = []
    for row in dense_matrix:
        string = []
        for val in row:
            string.append(_int_to_string(val, char_map_dict))
        string_list.append(''.join(s for s in string if s != '*'))
    return string_list

def _int_to_string(value, char_map_dict=None):
    if char_map_dict is None:
        char_map_dict = json.load(open(FLAGS.char_map_json_file, 'r'))
    assert(isinstance(char_map_dict, dict) and 'char_map_dict is not a dict')
    
    for key in char_map_dict.keys():
        if char_map_dict[key] == int(value):
            return str(key)
        elif len(char_map_dict.keys()) == int(value):
            return "" 
    raise ValueError('char map dict not has {:d} value. convert index to char failed.'.format(value))

def _inference_crnn_ctc():
    w, h = _IMAGE_SIZE
    input_images = tf.placeholder(dtype=tf.float32, shape=[1, h, w, 3])
    char_map_dict = json.load(open(FLAGS.char_map_json_file, 'r'))
    # initialise the net model
    crnn_net = model.CRNNCTCNetwork(phase='test',
                                    hidden_num=FLAGS.lstm_hidden_uints,
                                    layers_num=FLAGS.lstm_hidden_layers,
                                    num_classes=len(char_map_dict.keys()) + 1)

    with tf.variable_scope('CRNN_CTC', reuse=False):
        net_out = crnn_net.build_network(input_tensor=input_images)

    ctc_decoded, ct_log_prob = tf.nn.ctc_beam_search_decoder(net_out, _SEQUEENCE_LENGTH*np.ones(1), merge_repeated=True)

    with open(FLAGS.image_list, 'r') as fd:
       image_names = [line.strip() for line in fd.readlines()]

    # set checkpoint saver
    saver = tf.train.Saver()
    save_path = tf.train.latest_checkpoint(FLAGS.model_dir)

    with tf.Session() as sess:
        # restore all variables
        saver.restore(sess=sess, save_path=save_path)

        for image_name in image_names:
            image_path = os.path.join(FLAGS.image_dir, image_name)
            image = cv2.resize(cv2.imread(image_path), _IMAGE_SIZE)

            input_image = np.expand_dims(image, axis=0)
            
            preds = sess.run(ctc_decoded, feed_dict={input_images:input_image})
 
            preds = _sparse_matrix_to_list(preds[0])

            print('Predict {:s} image as: {:s}'.format(image_name, preds[0]))
        
def main(unused_argv):
    _inference_crnn_ctc()

if __name__ == '__main__':
    tf.app.run() 
