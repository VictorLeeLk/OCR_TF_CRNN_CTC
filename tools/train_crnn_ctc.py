from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import json

import tensorflow as tf

import numpy as np

from crnn_model import model

os.environ["CUDA_VISIBLE_DEVICES"]="0"

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
    'step_per_eval', 100, 'The number of training steps to run between evaluations.')

tf.app.flags.DEFINE_integer(
    'step_per_save', 1000, 'The number of training steps to run between save checkpoints.')

# ------------------------------------Basic prameters------------------------------------
tf.app.flags.DEFINE_integer(
    'batch_size', 32, 'The number of samples in each batch.')

tf.app.flags.DEFINE_integer(
    'max_train_steps', 20000, 'The number of maximum iteration steps for traning')

tf.app.flags.DEFINE_float(
    'learning_rate', 0.1, 'The initial learning rate for traning.')

tf.app.flags.DEFINE_integer(
    'decay_steps', 1000, 'The learning rate decay steps for traning.')

tf.app.flags.DEFINE_float(
    'decay_rate', 0.8, 'The learning rate decay rate for traning.')

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
    

def _train_crnn_ctc():
    tfrecord_path = os.path.join(FLAGS.data_dir, 'train.tfrecord')
    images, labels, _ = _read_tfrecord(tfrecord_path=tfrecord_path)

    # decode the training data from tfrecords
    input_images, input_labels = tf.train.shuffle_batch(
        tensors=[images, labels], batch_size=FLAGS.batch_size,
        capacity=1000 + 2*FLAGS.batch_size, min_after_dequeue=100, num_threads=FLAGS.num_threads)

    input_images = tf.cast(x=input_images, dtype=tf.float32)
    char_map_dict = json.load(open(FLAGS.char_map_json_file, 'r'))
    # initialise the net model
    crnn_net = model.CRNNCTCNetwork(phase='train',
                                    hidden_num=FLAGS.lstm_hidden_uints,
                                    layers_num=FLAGS.lstm_hidden_layers,
                                    num_classes=len(char_map_dict.keys()) + 1)

    with tf.variable_scope('CRNN_CTC', reuse=False):
        net_out = crnn_net.build_network(input_tensor=input_images)

    ctc_loss = tf.reduce_mean(tf.nn.ctc_loss(labels=input_labels, inputs=net_out, sequence_length=_SEQUEENCE_LENGTH*np.ones(FLAGS.batch_size)))

    ctc_decoded, ct_log_prob = tf.nn.ctc_beam_search_decoder(net_out, _SEQUEENCE_LENGTH*np.ones(FLAGS.batch_size), merge_repeated=False)

    sequence_distance = tf.reduce_mean(tf.edit_distance(tf.cast(ctc_decoded[0], tf.int32), input_labels))

    global_step = tf.train.create_global_step()

    learning_rate = tf.train.exponential_decay(FLAGS.learning_rate, global_step, FLAGS.decay_steps, FLAGS.decay_rate, staircase=True)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate).minimize(loss=ctc_loss, global_step=global_step)

    init_op = tf.global_variables_initializer()

    # set tf summary
    tf.summary.scalar(name='CTC_Loss', tensor=ctc_loss)
    tf.summary.scalar(name='Learning_Rate', tensor=learning_rate)
    tf.summary.scalar(name='Seqence_Distance', tensor=sequence_distance)
    merge_summary_op = tf.summary.merge_all()

    # set checkpoint saver
    saver = tf.train.Saver()
    if not os.path.exists(FLAGS.model_dir):
        os.makedirs(FLAGS.model_dir)
    train_start_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    model_name = 'crnn_ctc_ocr_{:s}.ckpt'.format(str(train_start_time))
    model_save_path = os.path.join(FLAGS.model_dir, model_name)  

    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    with tf.Session(config=sess_config) as sess:
        summary_writer = tf.summary.FileWriter(FLAGS.model_dir)
        summary_writer.add_graph(sess.graph)

        # init all variables
        sess.run(init_op)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        for step in range(FLAGS.max_train_steps):
            _, cl, lr, sd, preds, gt_labels, summary = sess.run(
                [optimizer, ctc_loss, learning_rate, sequence_distance, ctc_decoded, input_labels, merge_summary_op])

            if (step + 1) % FLAGS.step_per_save == 0: 
                summary_writer.add_summary(summary=summary, global_step=step)
                saver.save(sess=sess, save_path=model_save_path, global_step=step)

            if (step + 1) % FLAGS.step_per_eval == 0:
                # calculate the precision
                preds = _sparse_matrix_to_list(preds[0])
                gt_labels = _sparse_matrix_to_list(gt_labels)

                accuracy = []

                for index, gt_label in enumerate(gt_labels):
                    pred = preds[index]
                    total_count = len(gt_label)
                    correct_count = 0
                    try:
                        for i, tmp in enumerate(gt_label):
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
                accuracy = np.mean(np.array(accuracy).astype(np.float32), axis=0)

                print('step:{:d} learning_rate={:9f} ctc_loss={:9f} sequence_distance={:9f} train_accuracy={:9f}'.format(
                    step + 1, lr, cl, sd, accuracy))
            
        # close tensorboard writer
        summary_writer.close()

        # stop file queue
        coord.request_stop()
        coord.join(threads=threads)

def main(unused_argv):
    _train_crnn_ctc()

if __name__ == '__main__':
    tf.app.run() 
