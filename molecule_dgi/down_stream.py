import os
import time
import sys

import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
plt.switch_backend('agg')

from lib.fine_tune import Down_stream
from utils import preprocess_inputs
from utils import np_sigmoid
from utils import print_metrics

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
flags = tf.app.flags
FLAGS = flags.FLAGS

def read_data(file_name):
    f = open(file_name, 'r')
    lines = f.readlines()
    smi_list = []
    label_list = []
    for l in lines:
        smi = l.split(',')[0]
        label = float(l.split(',')[1].strip())
        smi_list.append(smi)
        label_list.append(label)
    return smi_list, label_list        

def train_pred(sess, opt_dict):

    def train_pred_one_epoch(sess, opt_dict, smi_list, label_list):
        num = 0
        loss_total = 0.0
        y_truth_total = np.empty([0,])
        y_pred_total = np.empty([0,])
        num_batches = len(smi_list) // FLAGS.batch_size
        if(len(smi_list)%FLAGS.batch_size != 0):
            num_batches += 1
        for i in range(num_batches):
            num += 1
            st_i = time.time()
            adj, x, y = preprocess_inputs(smi_list[i*FLAGS.batch_size:(i+1)*FLAGS.batch_size], 
                                          label_list[i*FLAGS.batch_size:(i+1)*FLAGS.batch_size],
                                          FLAGS.num_max_atoms) 

            feed_dict = {opt_dict.x:x, opt_dict.adj:adj, 
                         opt_dict.y:y, opt_dict.is_training:True}
            operations = []
            if FLAGS.optimize=='fully':             
                operations.append(opt_dict.train_fully)
            elif FLAGS.optimize=='predictor':    
                operations.append(opt_dict.train_pred)
            operations.append(opt_dict.logits)
            operations.append(opt_dict.pred_loss)

            _, y_pred, loss = sess.run(operations, feed_dict)
            y_pred = np_sigmoid(y_pred[:,0])

            loss_total += loss
            et_i = time.time()
            print ("Train_iter : ", num, \
                   ", loss :  ", loss, \
                "\t Time:", round(et_i-st_i,3))

            y_truth_total = np.concatenate((y_truth_total, y), axis=0)
            y_pred_total = np.concatenate((y_pred_total, y_pred), axis=0)

        loss_total /= num
        return loss_total, y_truth_total, y_pred_total

    def eval_pred_one_epoch(sess, opt_dict, smi_list, label_list):
        num = 0
        loss_total = 0.0
        y_truth_total = np.empty([0,])
        y_pred_total = np.empty([0,])
        num_batches = len(smi_list) // FLAGS.batch_size
        if(len(smi_list)%FLAGS.batch_size != 0):
            num_batches += 1

        for i in range(num_batches):
            num += 1
            st_i = time.time()
            adj, x, y = preprocess_inputs(smi_list[i*FLAGS.batch_size:(i+1)*FLAGS.batch_size], 
                                          label_list[i*FLAGS.batch_size:(i+1)*FLAGS.batch_size],
                                          FLAGS.num_max_atoms) 

            feed_dict = {opt_dict.x:x, opt_dict.adj:adj, 
                         opt_dict.y:y, opt_dict.is_training:False}
            y_pred, loss = sess.run(
                [opt_dict.logits, opt_dict.pred_loss], feed_dict=feed_dict)
            y_pred = np_sigmoid(y_pred[:,0])

            loss_total += loss
            et_i = time.time()

            y_truth_total = np.concatenate((y_truth_total, y), axis=0)
            y_pred_total = np.concatenate((y_pred_total, y_pred), axis=0)

        loss_total /= num
        return loss_total, y_truth_total, y_pred_total

    total_st = time.time()

    file_name = './data/'+FLAGS.prop+'.txt'
    smi_total, label_total = read_data(file_name)

    num_train = int(len(smi_total)*0.8)

    smi_test = smi_total[num_train:]
    label_test = label_total[num_train:]

    smi_train = smi_total[:int(num_train*FLAGS.train_ratio)]
    label_train = label_total[:int(num_train*FLAGS.train_ratio)]

    final_epoch = 0
    for epoch in range(FLAGS.num_epoches):
        print (epoch, "-th epoch")
        st = time.time()
        lr = FLAGS.lr * FLAGS.decay_rate**(epoch//FLAGS.decay_step)
        sess.run(tf.assign(opt_dict.lr, lr))

        total_train = list(zip(smi_train, label_train))
        random.shuffle(total_train)
        smi_train, label_train = [[x for x, y in total_train],
                                  [y for x, y in total_train]]

        # Training
        train_loss, y_truth_train, y_pred_train = \
            train_pred_one_epoch(sess, opt_dict, smi_train, label_train)

        # Validation
        test_loss, y_truth_test, y_pred_test = \
            eval_pred_one_epoch(sess, opt_dict, smi_test, label_test)

        # Print Results
        et = time.time()
        print ("Time for", epoch, "-th epoch: ", et-st)
        print ("Loss        Train:", round(train_loss,3),
               "\t Test:", round(test_loss,3))
        print_metrics(y_truth_train, y_pred_train)
        print_metrics(y_truth_test, y_pred_test)

        # Save network! 
        #if(FLAGS.save_model):
        #    saver.save(sess, ckpt_path, global_step=epoch)
        final_epoch=epoch    

    #saver.save(sess, ckpt_path, global_step=final_epoch)
    total_et = time.time()
    print ("Finish training! Total required time for training : ", (total_et-total_st))
    return


def main():
    # Construct the model 
    model = Down_stream(FLAGS)
    pre_trained = 'DGI'
    pre_trained += '_' + FLAGS.encoder
    pre_trained += '_' + FLAGS.pretrain_method
    opt_dict = model.get_opt_dict(pre_trained)
    print ("Start training")
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        st = time.time()    
        train_pred(sess, opt_dict)
        et = time.time()    
        print ("Successfully training a predictor. Time for training:", et-st)
    return

if __name__ == '__main__':
    # Hyperparameters for a graph convolutional encoder, a readout and a predictor
    flags.DEFINE_integer('seed', 999, 'random seed')
    flags.DEFINE_integer('num_max_atoms', 100, 'Maximum number of atoms in a single molecule')
    flags.DEFINE_integer('hidden_dim', 64, 'output dimension of graph convolution layers')
    flags.DEFINE_integer('latent_dim', 256, 'output dimension of readout and mlp layers')
    flags.DEFINE_integer('num_layers', 4, 'number of hidden layers')
    flags.DEFINE_integer('num_attn', 4, 'number of heads for attentions')
    flags.DEFINE_float('dropout', 0.0, 'Dropout probability')
    flags.DEFINE_float('weight_decay', 0.0, 'length scale of gaussian prior for l2-regularizations')
    flags.DEFINE_bool('normalization', True, ', options: batch_norm, layer_norm, instance_norm')
    flags.DEFINE_string('encoder', 'gcn', 'gcn/gat')

    # Hyperparameters for a optimization process
    flags.DEFINE_string('prop', 'BACE', 'BACE, BBBP, HIV, Tox21, ...')
    flags.DEFINE_float('train_ratio', 1.0, 'Initial learning rate')
    flags.DEFINE_float('lr', 1e-4, 'Initial learning rate')
    flags.DEFINE_float('decay_rate', 1.0, 'Initial learning rate')
    flags.DEFINE_integer('decay_step', 10, 'Initial learning rate')
    flags.DEFINE_integer('batch_size', 100, 'Batch size')
    flags.DEFINE_integer('num_epoches', 200, 'Epoch size')
    flags.DEFINE_bool('save_model', False, '')
    flags.DEFINE_string('pretrain_method', 'wpc', 'options: info_nce, jsd')
    flags.DEFINE_bool('use_pretrained', True, 'Whether to use pre-trained encoders')
    flags.DEFINE_string('optimize', 'fully', 'Optimize only predictor or both predictor and encoder')

    main()
