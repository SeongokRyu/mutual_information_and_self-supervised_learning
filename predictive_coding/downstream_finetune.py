import os
import glob
import time

import random
import numpy as np
import tensorflow as tf

from libs.finetune import Finetune
from libs.dataset import read_file_list
from libs.dataset import read_speech_file
from libs.dataset import read_labels_txt
from libs.dataset import read_splited_speech_file
from libs.dataset import get_speaker_label
from sklearn.metrics import accuracy_score

#os.environ["CUDA_VISIBLE_DEVICES"] = "3"
flags = tf.app.flags
FLAGS = flags.FLAGS

def train(sess, ops_dict, pre_trained):

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
            y_pred_total = np.concatenate((y_pred_total, pred_batch), axis=0)

            total_loss += loss

        total_loss /= num_batches
        print ("Finish a single iteration, Total training loss of the entire batch:", total_loss)
        return total_loss, y_true_total, y_pred_total


    random.seed(FLAGS.seed)
    total_st = time.time()

    model_name = 'LibriSpeech'
    model_name += '_' + FLAGS.task
    model_name += '_transfer' 
    model_name += '_'+FLAGS.method 
    model_name += '_' + str(FLAGS.train_ratio)
    model_name += '_' + str(FLAGS.seed)
    ckpt_path = './save/'+model_name+'.ckpt'

    idx_gender_speaker = read_labels_txt(FLAGS.path)
    train_file_list = read_splited_speech_file(FLAGS.path, train_or_test='train')
    test_file_list = read_splited_speech_file(FLAGS.path, train_or_test='test')

    random.shuffle(train_file_list)
    num_train = int(FLAGS.train_ratio*len(train_file_list))
    train_file_list = train_file_list[:num_train]

    saver = tf.train.Saver()
    final_epoch = 0
    for epoch in range(1, FLAGS.num_epoches+1):
        print (epoch, "-th epoch")
        st = time.time()

        random.shuffle(train_file_list)

        train_loss, label_train, pred_train = run_one_epoch(sess, ops_dict, train_file_list, 
                                                            idx_gender_speaker, True)
        train_accuracy = accuracy_score(
            label_train.astype(int), pred_train.astype(int))
        print ("Train           Loss:", round(train_loss,3), \
                            "\t Accuracy:", round(train_accuracy,3))

        if epoch%FLAGS.test_step == 0:
            test_loss, label_test, pred_test = run_one_epoch(sess, ops_dict, test_file_list, 
                                                             idx_gender_speaker, False)
            test_accuracy = accuracy_score(
                label_test.astype(int), pred_test.astype(int))
            print ("Test            Loss:", round(test_loss,3), \
                                "\t Accuracy:", round(test_accuracy,3))

        et = time.time()
        print ("Time for", epoch, "-th epoch: ", et-st)

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
    pre_trained = 'LibriSpeech'
    pre_trained += '_' + 'pretrain'
    pre_trained += '_' + FLAGS.method
    pre_trained += '_re'
    pre_trained += '.ckpt' 

    idx_list = []
    file_list = glob.glob('./save/'+pre_trained+'*.meta')
    for file_name in file_list:
        idx = int(file_name.split('-')[1].split('.')[0])
        idx_list.append(idx)
    last_idx = max(idx_list)   
    pre_trained += '-'+str(last_idx)

    num_categories = {'gender':1, 'speaker':251}
    model = Finetune(window=FLAGS.window, 
                     conv_dim=FLAGS.conv_dim,
                     ar_dim=FLAGS.ar_dim,
                     init_lr=FLAGS.init_lr,
                     num_categories=num_categories[FLAGS.task],
                     pre_trained=pre_trained)

    ops_dict = model.ops_dict()

    print ("Start training")
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        st = time.time()    
        train(sess, ops_dict, pre_trained)
        et = time.time()    
        print ("Finishi fine-tuning. Time for training:", et-st)
    return

if __name__ == '__main__':
    flags.DEFINE_integer('seed', 999, 'random seed')
    flags.DEFINE_string('path', './data/LibriSpeech/', '')
    flags.DEFINE_string('task', 'speaker', 'speaker or gender classification')
    flags.DEFINE_integer('window', 20480, 'Output dimension of a convolutional encoder')
    flags.DEFINE_integer('conv_dim', 512, 'Output dimension of a convolutional encoder')
    flags.DEFINE_integer('ar_dim', 256, 'Output dimension of a GRU-RNN')
    flags.DEFINE_float('init_lr', 2e-4, 'Initial learning rate')
    flags.DEFINE_float('train_ratio', 1.0, 'the ratio of training data')
    flags.DEFINE_integer('batch_size', 16, 'Batch size')
    flags.DEFINE_integer('num_epoches', 100, 'numer of future steps to predict')
    flags.DEFINE_integer('test_step', 10, 'the index of last checkpoint file, if it exists')
    flags.DEFINE_bool('save_model', False, 'Whether to save models during training procedures')
    flags.DEFINE_integer('last_ckpt', -1, 'the index of last checkpoint file, if it exists')
    # For loading a pre-trained model
    flags.DEFINE_string('method', 'WPC', 'the method used for pre-training')

    main()
