import os
import time

import random
import numpy as np
import tensorflow as tf

from libs.supervised import Supervised
from libs.dataset import read_file_list
from libs.dataset import read_speech_file
from libs.dataset import read_labels_txt
from libs.dataset import read_splited_speech_file
from libs.dataset import get_speaker_label


os.environ["CUDA_VISIBLE_DEVICES"] = "3"
flags = tf.app.flags
FLAGS = flags.FLAGS

def train(sess, ops_dict):

    def run_one_epoch(sess, ops_dict, file_list, idx_gender_speaker, is_train):
        num_files = len(file_list)
        num_batches = num_files // FLAGS.batch_size
        if len(file_list)%FLAGS.batch_size != 0:
            num_batches += 1
        print ("Number of examples:", num_files, "Number of batches", num_batches)
        
        label = {'gender':0, 'speaker':1}
        pred_ops = {'gender':ops_dict.pred_binary, 
                    'speaker':ops_dict.pred_softmax}

        total_loss = 0.0
        y_true_total = np.empty([0,])
        y_pred_total = np.empty([0,])
        for i in range(num_batches):
            st_i = time.time()
            file_batch = file_list[i*FLAGS.batch_size:(i+1)*FLAGS.batch_size]
            x_batch = []
            y_batch = []
            for file_path in file_batch:
                x_batch.append(
                    np.load(file_path))
                y_batch.append(
                    get_speaker_label(idx_gender_speaker, file_path, label[FLAGS.task]))
            x_batch = np.asarray(x_batch)    
            y_batch = np.asarray(y_batch)    

            loss = 0.0
            feed_dict = {ops_dict.x:x_batch, ops_dict.y:y_batch}
            ops = []
            if is_train:
                ops = [ops_dict.train_op, ops_dict.loss, pred_ops[FLAGS.task]]
                _, loss, pred_batch = sess.run(ops, feed_dict)
                et_i = time.time()
                print (i, "-th batch, \t Loss:", loss, \
                       "\t Time:", round(et_i-st_i, 3))    
            else:
                ops = [ops_dict.loss, pred_ops[FLAGS.task]]
                loss, pred_batch = sess.run(ops, feed_dict)

            if FLAGS.task=='speaker':
                pred_batch = np.argmax(pred_batch, axis=1)
            y_true_total = np.concatenate((y_true_total, y_batch), axis=0)
            y_pred_total = np.concatenate((y_true_total, pred_batch), axis=0)

            total_loss += loss

        total_loss /= num_batches
        print ("Finish a single iteration, Total training loss of the entire batch:", total_loss)
        return total_loss, y_true_total, y_pred_total

    total_st = time.time()

    model_name = 'LibriSpeech'
    model_name += '_supervised' 
    model_name += '_' + FLAGS.task

    idx_gender_speaker = read_labels_txt(FLAGS.path)
    train_file_list = read_splited_speech_file(FLAGS.path, train_or_test='train')
    test_file_list = read_splited_speech_file(FLAGS.path, train_or_test='test')

    saver = tf.train.Saver()
    ckpt_path = './save/'+model_name+'.ckpt'
    final_epoch = 0
    for epoch in range(FLAGS.num_epoches):
        print (epoch, "-th epoch")
        st = time.time()

        random.shuffle(train_file_list)

        # Training
        train_loss, label_train, pred_train = run_one_epoch(sess, ops_dict, train_file_list, 
                                                            idx_gender_speaker, True)

        # Validation
        #test_loss, label_test, pred_test = run_one_epoch(sess, ops_dict, test_file_list, 
        #                                                 idx_gender_speaker, False)

        # Print Results
        et = time.time()
        print ("Time for", epoch, "-th epoch: ", et-st)
        print ("Loss        Train:", round(train_loss,3))
              # "\t Test:", round(validation_loss,3))

        # Save network 
        if(FLAGS.save_model):
            saver.save(sess, ckpt_path, global_step=epoch)
        final_epoch=epoch    

    saver.save(sess, ckpt_path, global_step=final_epoch)
    total_et = time.time()
    print ("Finish training! Total required time for training : ", (total_et-total_st))
    return

def main():
    # Construct the model 
    num_categories = {'gender':1, 'speaker':251}
    model = Supervised(window=FLAGS.window, 
                       conv_dim=FLAGS.conv_dim,
                       ar_dim=FLAGS.ar_dim,
                       init_lr=FLAGS.init_lr,
                       num_categories=num_categories[FLAGS.task])

    ops_dict = model.ops_dict()

    print ("Start training")
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        st = time.time()    
        train(sess, ops_dict)
        et = time.time()    
        print ("Finishi pre-training. Time for training:", et-st)
    return

if __name__ == '__main__':
    flags.DEFINE_string('path', './data/LibriSpeech/', '')
    flags.DEFINE_string('task', 'speaker', 'speaker or gender classification')
    flags.DEFINE_integer('window', 20480, 'Output dimension of a convolutional encoder')
    flags.DEFINE_integer('conv_dim', 512, 'Output dimension of a convolutional encoder')
    flags.DEFINE_integer('ar_dim', 256, 'Output dimension of a GRU-RNN')
    flags.DEFINE_float('init_lr', 2e-4, 'Initial learning rate')
    flags.DEFINE_integer('batch_size', 16, 'Batch size')
    flags.DEFINE_integer('num_epoches', 100, 'numer of future steps to predict')
    flags.DEFINE_bool('save_model', False, 'Whether to save models during training procedures')

    main()
