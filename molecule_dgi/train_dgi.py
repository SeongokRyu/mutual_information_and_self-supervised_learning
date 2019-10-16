import os
import time
import sys

import glob
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
plt.switch_backend('agg')

from lib.dgi import DGI
from utils import convert_to_graph
from utils import np_sigmoid
from utils import print_metrics

#os.environ["CUDA_VISIBLE_DEVICES"] = "3"
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

def read_all_smiles(path):

    def read_single_file(file_name):
        f = open(file_name, 'r')
        lines = f.readlines()
        smi_list = []
        for l in lines:
            smi = l.split(',')[0].strip()
            smi_list.append(smi)
        return smi_list    

    file_list = glob.glob(path+'*.txt')
    smi_all = []
    for file_name in file_list:
        smi_list = read_single_file(file_name)
        smi_all += smi_list

    return smi_all
    

def train_dgi(sess, opt_dict):

    def train_dgi_one_epoch(sess, opt_dict, smi_list):
        num = 0
        loss_total = 0.0
        num_batches = len(smi_list) // FLAGS.batch_size
        if(len(smi_list)%FLAGS.batch_size != 0):
            num_batches += 1

        for i in range(num_batches):
            num += 1
            st_i = time.time()
            adj, x = convert_to_graph(smi_list[i*FLAGS.batch_size:(i+1)*FLAGS.batch_size], 
                                            FLAGS.num_max_atoms) 
            feed_dict = {opt_dict.x:x, opt_dict.adj:adj}
            _, loss = sess.run(
                [opt_dict.train_dgi, opt_dict.dgi_loss], feed_dict=feed_dict)
            loss_total += loss
            et_i = time.time()
            print ("Train_iter : ", num, \
                   ", loss :  ", loss, \
                "\t Time:", round(et_i-st_i,3))

        loss_total /= num
        return loss_total

    def eval_dgi_one_epoch(sess, opt_dict, smi_list):
        num = 0
        loss_total = 0.0

        num_batches = len(smi_list) // FLAGS.batch_size
        if(len(smi_list)%FLAGS.batch_size != 0):
            num_batches += 1

        for i in range(num_batches):
            num += 1
            st_i = time.time()
            adj, x = convert_to_graph(smi_list[i*FLAGS.batch_size:(i+1)*FLAGS.batch_size], 
                                                FLAGS.num_max_atoms) 
            feed_dict = {opt_dict.x:x, opt_dict.adj:adj}
            loss = sess.run(opt_dict.dgi_loss, feed_dict=feed_dict)

            loss_total += loss
            et_i = time.time()

        loss_total /= num
        return loss_total

    random.seed(FLAGS.seed)

    model_name = 'DGI'
    model_name += '_' + FLAGS.encoder
    model_name += '_' + FLAGS.mi_loss
    total_st = time.time()
    smi_total = read_all_smiles('./data/')

    num_train= int(len(smi_total)*1.0)
    smi_train = smi_total[:num_train]
    #smi_validation = smi_total[num_train:]

    print ("Number of SMILES", num_train)
    print ("Number of training batches", int(num_train//FLAGS.batch_size)+1)

    saver = tf.train.Saver()
    ckpt_path = './save/'+model_name+'.ckpt'
    final_epoch = 0
    for epoch in range(FLAGS.num_epoches):
        print (epoch, "-th epoch")
        st = time.time()
        lr = FLAGS.lr
        sess.run(tf.assign(opt_dict.lr, lr))

        random.shuffle(smi_train)

        # Training
        train_loss = train_dgi_one_epoch(sess, opt_dict, smi_train)

        # Validation
        # validation_loss = eval_dgi_one_epoch(sess, opt_dict, smi_validation)

        # Print Results
        et = time.time()
        print ("Time for", epoch, "-th epoch: ", et-st)
        print ("Loss        Train:", round(train_loss,3))

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
    model = DGI(FLAGS)
    opt_dict = model.get_opt_dict()
    print ("Start training")
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        st = time.time()    
        train_dgi(sess, opt_dict)
        et = time.time()    
        print ("Successfully training an encoder. Time for training:", et-st)
    return

if __name__ == '__main__':
    # Hyperparameters for a graph convolutional encoder, a readout and a predictor
    flags.DEFINE_integer('seed', 999, 'random seed')
    flags.DEFINE_string('encoder', 'gcn', 'options: gcn, gat')
    flags.DEFINE_integer('num_max_atoms', 100, 'Maximum number of atoms in a single molecule')
    flags.DEFINE_integer('hidden_dim', 64, 'output dimension of graph convolution layers')
    flags.DEFINE_integer('latent_dim', 256, 'output dimension of readout and mlp layers')
    flags.DEFINE_integer('num_layers', 4, 'number of hidden layers')
    flags.DEFINE_integer('num_attn', 4, 'number of heads for a self-attention mechanism')
    flags.DEFINE_float('dropout', 0.1, 'Dropout probability, e.g. 0.1 means 10% of hidden layers will be dropped out')
    flags.DEFINE_bool('normalization', True, ', options: batch_norm, layer_norm, instance_norm')
    flags.DEFINE_string('mi_loss', 'wpc', 'options: jsd, kld, cpc, wpc')

    # Hyperparameters for a optimization process
    flags.DEFINE_float('lr', 2e-4, 'Initial learning rate')
    flags.DEFINE_float('decay_rate', 1.0, 'Initial learning rate')
    flags.DEFINE_integer('decay_step', 10, 'Initial learning rate')
    flags.DEFINE_integer('batch_size', 100, 'Batch size')
    flags.DEFINE_integer('num_epoches', 50, 'Epoch size')
    flags.DEFINE_bool('save_model', True, '')
    main()
