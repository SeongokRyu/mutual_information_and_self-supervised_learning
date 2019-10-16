import collections
import re
import glob
import numpy as np
import tensorflow as tf

from easydict import EasyDict

from lib.blocks import encoder_gcn
from lib.blocks import encoder_gcn_attn
from lib.blocks import readout_atomwise
from lib.blocks import linear_predictor

class Down_stream():
    def __init__(self, FLAGS):
        self.FLAGS = FLAGS

    def get_opt_dict(self, pre_trained=None, **kwargs):
        del kwargs

        # Placeholders
        n_atoms = self.FLAGS.num_max_atoms
        adj = tf.placeholder(tf.float32, shape = [None, n_atoms, n_atoms])
        x = tf.placeholder(tf.float32, shape = [None, n_atoms, 58])
        y = tf.placeholder(tf.float32, shape = [None])
        is_training = tf.placeholder(tf.bool, shape = ())

        with tf.variable_scope('encoder_parameters'):
            encoder = {'gcn':encoder_gcn, 'gat':encoder_gcn_attn}
            node_feature = encoder[self.FLAGS.encoder](adj, x, is_training, self.FLAGS)
            graph_feature = readout_atomwise(node_feature, self.FLAGS)

        with tf.variable_scope('predictor_parameters'):       
            logits = linear_predictor(graph_feature)

        pred_loss = self.bce_loss(y, logits)

        # Variables
        var_total = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        var_encoder = [v for v in var_total if 'encoder_parameters' in v.name]
        var_predictor = [v for v in var_total if 'predictor_parameters' in v.name]  
        var_decay = [v for v in var_predictor if 'kernel' in v.name]

        if self.FLAGS.use_pretrained:
            idx_list = []
            file_list = glob.glob('./save/'+pre_trained+'*.meta')
            for file_name in file_list:
                idx = int(file_name.split('-')[1].split('.')[0])
                idx_list.append(idx)
            last_idx = max(idx_list)   

            init_checkpoint = './save/'+pre_trained+'.ckpt-'+str(last_idx)
            assignment_map = self.get_assignment_map_from_checkpoint(
                var_encoder, init_checkpoint)
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        # Optimizer            
        lr = tf.Variable(self.FLAGS.lr, trainable=False)
        optimizer = tf.contrib.opt.AdamWOptimizer(
            weight_decay=self.FLAGS.weight_decay, learning_rate=lr)

        train_pred = optimizer.minimize(loss=pred_loss, 
                                       var_list=var_predictor, 
                                       decay_var_list=var_decay)
        train_fully = optimizer.minimize(loss=pred_loss,
                                       var_list=var_total,
                                       decay_var_list=var_decay)
        print ("Complete preparing a computational graph")
        return EasyDict(
            x=x, adj=adj, y=y, is_training=is_training, lr=lr, 
            logits=logits, graph_feature=graph_feature, pred_loss=pred_loss, 
            train_pred=train_pred, train_fully=train_fully)
            
    
    def get_assignment_map_from_checkpoint(self, trainable_variables, init_checkpoint):
        assignment_map = {}
        name_to_variable = collections.OrderedDict()
        for var in trainable_variables:
            name = var.name
            m = re.match("^(.*):\\d+$", name)
            if m is not None:
                name = m.group(1)
            name_to_variable[name] = var

        init_variables = tf.train.list_variables(init_checkpoint)
        for x in init_variables:
            (name, var) = (x[0], x[1])
            if name not in name_to_variable:
                continue
            assignment_map[name] = name_to_variable[name]
            #assignment_map[name] = name
        return assignment_map

    def bce_loss(self, y, logits):
        nll = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits[:,0]))
        return nll
