from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import json

import tensorflow as tf

import numpy as np

from crnn_model import model

os.environ["CUDA_VISIBLE_DEVICES"]=""

_IMAGE_SIZE = (100, 32)
_SEQUEENCE_LENGTH = 25 # 100 /4

# ------------------------------------Basic prameters------------------------------------
tf.app.flags.DEFINE_string(
    'data_dir', './tfrecords/', 'Path to the directory containing data tf record.')

tf.app.flags.DEFINE_string(
    'model_dir', './model/', 'Base directory for the model.')

tf.app.flags.DEFINE_integer(
    'num_threads', 4, 'The number of threads to use in batch shuffling') 

tf.app.flags.DEFINE_integer(
    'batch_size', 32, 'The number of samples in each batch.')

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
    assert(isinstance(char_map_dict, dict))    

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
    assert(isinstance(char_map_dict, dict))
    
    for key in char_map_dict.keys():
        if char_map_dict[key] == int(value):
            return str(key)
        elif len(char_map_dict.keys()) == int(value):
            return "" 
    raise ValueError('char map dict not has {:d} value. convert index to char failed.'.format(value))

def _read_tfrecord(tfrecord_path, num_epochs=None):
    if not os.path.exists(tfrecord_path):
        raise ValueError('cannott find tfrecord file in path: {:s}'.format(tfrecord_path))

    filename_queue = tf.train.string_input_producer([tfrecord_path], num_epochs=num_epochs)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'images': tf.FixedLenFeature([], tf.string),
                                           'labels': tf.VarLenFeature(tf.int64),
                                           'imagenames': tf.FixedLenFeature([], tf.string),
                                       })
    images = tf.decode_raw(features['images'], tf.uint8)
    w, h = _IMAGE_SIZE 
    images = tf.reshape(images, [h, w, 3])
    labels = features['labels']
    labels = tf.cast(labels, tf.int32)
    imagenames = features['imagenames']
    return images, labels, imagenames
    

def _eval_crnn_ctc():
    tfrecord_path = os.path.join(FLAGS.data_dir, 'validation.tfrecord')
    images, labels, imagenames = _read_tfrecord(tfrecord_path=tfrecord_path)

    # decode the training data from tfrecords
    input_images, input_labels, input_names = tf.train.batch(
        tensors=[images, labels, imagenames], batch_size=FLAGS.batch_size,
        capacity=1000 + 2*FLAGS.batch_size, num_threads=FLAGS.num_threads)

    input_images = tf.cast(x=input_images, dtype=tf.float32)
    char_map_dict = json.load(open(FLAGS.char_map_json_file, 'r'))
    # initialise the net model
    crnn_net = model.CRNNCTCNetwork(phase='test',
                                    hidden_num=FLAGS.lstm_hidden_uints,
                                    layers_num=FLAGS.lstm_hidden_layers,
                                    num_classes=len(char_map_dict.keys()) + 1)

    with tf.variable_scope('CRNN_CTC', reuse=False):
        net_out = crnn_net.build_network(input_tensor=input_images)

    ctc_decoded, ct_log_prob = tf.nn.ctc_beam_search_decoder(net_out, _SEQUEENCE_LENGTH*np.ones(FLAGS.batch_size), merge_repeated=False)

    # set checkpoint saver
    saver = tf.train.Saver()
    save_path = tf.train.latest_checkpoint(FLAGS.model_dir)

    test_sample_count = 0
    for record in tf.python_io.tf_record_iterator(tfrecord_path):
        test_sample_count += 1
    step_nums = test_sample_count // FLAGS.batch_size

    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    with tf.Session(config=sess_config) as sess:
        # restore all variables
        saver.restore(sess=sess, save_path=save_path)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        accuracy = []

        for _ in range(step_nums):
            preds, imgs, lbls, names = sess.run([ctc_decoded, input_images, input_labels, input_names])
 
            preds = _sparse_matrix_to_list(preds[0])
            lbls = _sparse_matrix_to_list(lbls)

            for index, lbl in enumerate(lbls):
                pred = preds[index]
                total_count = len(lbl)
                correct_count = 0
                try:
                    for i, tmp in enumerate(lbl):
                        if tmp == pred[i]:
                            correct_count += 1
                except IndexError:
                    continue
                finally:
                    try:
                        accuracy.append(correct_count / total_count)
                    except ZeroDivisionError:
                        if len(pred) == 0:
                            accuracy.append(1)
                        else:
                            accuracy.append(0)

            for index, img in enumerate(imgs):
                print('Predict {:s} image with gt label: {:s} <--> predict label: {:s}'.format(names[index], lbls[index], preds[index]))
        
        accuracy = np.mean(np.array(accuracy).astype(np.float32), axis=0)
        print('Mean test accuracy is {:5f}'.format(accuracy))

        # stop file queue
        coord.request_stop()
        coord.join(threads=threads)

def main(unused_argv):
    _eval_crnn_ctc()

if __name__ == '__main__':
    tf.app.run() 
