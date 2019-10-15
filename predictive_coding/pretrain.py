import os
import time

import random
import numpy as np
import tensorflow as tf

from libs.self_supervised import SelfSupervised
from libs.dataset import read_file_list
from libs.dataset import read_speech_file
from libs.dataset import read_splited_speech_file

#os.environ["CUDA_VISIBLE_DEVICES"] = "3"
flags = tf.app.flags
FLAGS = flags.FLAGS

def train(sess, ops_dict):

    def run_one_epoch(sess, ops_dict, file_list, is_train):
        num_files = len(file_list)
        num_batches = num_files // FLAGS.batch_size
        if len(file_list)%FLAGS.batch_size != 0:
            num_batches += 1
        print ("Number of examples:", num_files, "Number of batches", num_batches)

        total_loss = 0.0
        for i in range(num_batches):
            st_i = time.time()
            file_batch = file_list[i*FLAGS.batch_size:(i+1)*FLAGS.batch_size]
            x_batch = []
            for file_path in file_batch:
                x_batch.append(np.load(file_path))
                #x_batch.append(np.load(FLAGS.path+'./train-clean-100/train_split/'+file_name))
            x_batch = np.asarray(x_batch)    

            feed_dict = {ops_dict.x:x_batch}
            ops = []
            loss = 0.0
            if is_train:
                ops = [ops_dict.train_op, ops_dict.obj]
                _, loss = sess.run(ops, feed_dict)
                et_i = time.time()
                print (i, "-th batch, \t Loss:", loss, \
                       "Time for training:", round(et_i-st_i, 3))    
            else:
                ops = ops_dict.obj
                loss = sess.run(ops, feed_dict)
            total_loss += loss
        total_loss /= num_batches

        print ("Finish a single iteration, Total training loss of the entire batch:", total_loss)
        return total_loss

    total_st = time.time()

    model_name = 'LibriSpeech'
    model_name += '_pretrain' 
    method = 'CPC'
    if FLAGS.use_lipschitz:
        method = 'WPC'
    model_name += '_' + method
    model_name += '_re'

    train_file_list = read_splited_speech_file(FLAGS.path, train_or_test='train')
    #test_file_list = read_splited_speech_file(FLAGS.path, train_or_test='test')

    saver = tf.train.Saver()
    ckpt_path = './save/'+model_name+'.ckpt'
    if FLAGS.last_ckpt != -1:
        saver.restore(sess, ckpt_path+'-'+str(FLAGS.last_ckpt))
        print ("Restart training by reloading the", FLAGS.last_ckpt, "-th ckpt file")

    final_epoch = 0
    for epoch in range(FLAGS.last_ckpt+1, FLAGS.num_epoches):
        print (epoch, "-th epoch")
        st = time.time()

        random.shuffle(train_file_list)

        # Training
        train_loss = run_one_epoch(sess, ops_dict, train_file_list, True)

        # Validation
        #test_loss = run_one_epoch(sess, ops_dict, test_file_list, False)

        # Print Results
        et = time.time()
        print ("Time for", epoch, "-th epoch: ", et-st)
        print ("Loss        Train:", round(train_loss,3))
              # "\t Validation:", round(validation_loss,3))

        # Save network! 
        if(FLAGS.save_model):
            saver.save(sess, ckpt_path, global_step=epoch)
        final_epoch=epoch    

    saver.save(sess, ckpt_path, global_step=final_epoch)
    total_et = time.time()
    print ("Finish training! Total required time for training : ", (total_et-total_st))
    return

def main():
    # Construct the model 
    model = SelfSupervised(window=FLAGS.window, 
                           conv_dim=FLAGS.conv_dim,
                           ar_dim=FLAGS.ar_dim,
                           init_lr=FLAGS.init_lr,
                           num_future_steps=FLAGS.num_future_steps,
                           use_lipschitz=FLAGS.use_lipschitz)
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
    flags.DEFINE_integer('window', 20480, 'Output dimension of a convolutional encoder')
    flags.DEFINE_integer('conv_dim', 512, 'Output dimension of a convolutional encoder')
    flags.DEFINE_integer('ar_dim', 256, 'Output dimension of a GRU-RNN')
    flags.DEFINE_integer('num_future_steps', 12, 'numer of future steps to predict')
    flags.DEFINE_float('init_lr', 2e-4, 'Initial learning rate')
    flags.DEFINE_integer('batch_size', 16, 'Batch size')
    flags.DEFINE_integer('num_epoches', 30, 'numer of future steps to predict')
    flags.DEFINE_bool('use_lipschitz', False, 'Whether to enforce 1-Lipscthiz condition')
    flags.DEFINE_bool('save_model', True, 'Whether to save models during training procedures')
    flags.DEFINE_integer('last_ckpt', -1, 'the index of last checkpoint file, if it exists')

    main()
