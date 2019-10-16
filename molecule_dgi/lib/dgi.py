import numpy as np
import tensorflow as tf

from easydict import EasyDict

from lib.blocks import encoder_gcn
from lib.blocks import encoder_gcn_attn
from lib.blocks import readout_atomwise
from lib.loss_functions import mi_js_divergence
from lib.loss_functions import mi_kl_divergence
from lib.loss_functions import mi_info_nce
from lib.utils import spectral_normalization

class DGI():
    def __init__(self, FLAGS):
        self.FLAGS = FLAGS

    def get_opt_dict(self, **kwargs):
        del kwargs

        # Placeholders
        n_atoms = self.FLAGS.num_max_atoms
        adj = tf.placeholder(tf.float32, shape = [None, n_atoms, n_atoms])
        x = tf.placeholder(tf.float32, shape = [None, n_atoms, 58])

        with tf.variable_scope('encoder_parameters'):
            encoder = {'gcn':encoder_gcn, 'gat':encoder_gcn_attn}
            node_feature = encoder[self.FLAGS.encoder](adj, x, True, self.FLAGS)
            graph_feature = readout_atomwise(node_feature, self.FLAGS)

            # weight parameter for the discriminator function
            kernel_d = tf.get_variable(name='kernel_d', 
                                       shape=[self.FLAGS.hidden_dim, self.FLAGS.latent_dim],
                                       initializer=tf.glorot_normal_initializer(),
                                       trainable=True)
            if self.FLAGS.mi_loss=='wpc':
                kernel_d = spectral_normalization(kernel_d, 1, 1) 

        dgi_loss = None
        if self.FLAGS.mi_loss=='jsd':
            dgi_loss = mi_js_divergence(node_feature, graph_feature, kernel_d)
        elif self.FLAGS.mi_loss=='kld':
            dgi_loss = mi_kl_divergence(node_feature, graph_feature, kernel_d)
        elif self.FLAGS.mi_loss=='cpc' or self.FLAGS.mi_loss=='wpc':
            dgi_loss = mi_info_nce(node_feature, graph_feature, kernel_d)
        else:
            dgi_loss = mi_info_nce(node_feature, graph_feature, kernel_d)

        # Variables
        var_total = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        var_enc = [v for v in var_total if 'encoder_parameters' in v.name]

        # Optimizer            
        lr = tf.Variable(self.FLAGS.lr, trainable=False)
        train_dgi = tf.train.AdamOptimizer(lr).minimize(loss=dgi_loss, 
                                                        var_list=var_enc)

        print ("Complete preparing a computational graph")
        return EasyDict(
            x=x, adj=adj, lr=lr, graph_feature=graph_feature, 
            dgi_loss=dgi_loss, train_dgi=train_dgi)
