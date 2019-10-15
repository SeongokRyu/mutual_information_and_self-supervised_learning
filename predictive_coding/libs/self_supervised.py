import tensorflow as tf
import functools
from easydict import EasyDict

from libs.blocks import voice_encoder
from libs.blocks import gru_rnn
from libs.objectives import info_nce
from libs.utils import spectral_normalization

class SelfSupervised():

    def __init__(self,
                 window=20480, 
                 conv_dim=512, 
                 ar_dim=256, 
                 init_lr=2e-4, 
                 num_future_steps=12,
                 use_lipschitz=False):

        self.window = window
        self.conv_dim = conv_dim
        self.ar_dim = ar_dim
        self.init_lr = init_lr
        self.num_future_steps = num_future_steps
        self.use_lipschitz = use_lipschitz

    def ops_dict(self, **kwargs):
        x = tf.placeholder(tf.float32, shape=[None, self.window, ])

        encoder = functools.partial(voice_encoder, **kwargs)
        rnn = functools.partial(gru_rnn, **kwargs)

        with tf.variable_scope("enc", reuse=False):
            z = encoder(tf.expand_dims(x, axis=-1), self.conv_dim)

        with tf.variable_scope("ar", reuse=False):
            context = rnn(z, self.ar_dim)


        with tf.variable_scope("obj", reuse=False):
            w_obj_list = []
            for j in range(self.num_future_steps):
                w_obj = tf.get_variable(name='kernel_obj'+str(j),
                                        shape=[self.ar_dim, self.conv_dim],
                                        dtype=tf.float32,
                                        initializer=tf.random_normal_initializer())
                # Contrastive Predictive Coding becomes Wasserstein Predictive Coding
                # by enforcing 1-Lipscthiz continuity on the MI-estimator.
                if self.use_lipschitz:
                    w_obj = spectral_normalization(w_obj, 1, j) 
                w_obj_list.append(w_obj)    

        var_total = tf.trainable_variables()
        obj_list = []
        for j in range(self.num_future_steps):
            obj = info_nce(w_obj_list[j], z, context, j+1, self.num_future_steps)
            obj_list.append(obj)
        total_obj = -tf.reduce_mean(tf.concat(obj_list, axis=-1))    

        lr = tf.Variable(self.init_lr, trainable=False)
        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        train_op = optimizer.minimize(loss=total_obj)

        return EasyDict(
            x=x, z=z, obj=total_obj, lr=lr, train_op=train_op)
