import tensorflow as tf
import functools
from easydict import EasyDict

from libs.blocks import voice_encoder
from libs.blocks import gru_rnn
from libs.blocks import linear_classifier
from libs.objectives import binary_ce_loss
from libs.objectives import softmax_ce_loss

class Supervised():

    def __init__(self,
                 window=20480, 
                 conv_dim=512, 
                 ar_dim=256, 
                 num_categories=251, # For speaker classification
                 init_lr=2e-4):

        self.window = window
        self.conv_dim = conv_dim
        self.ar_dim = ar_dim
        self.init_lr = init_lr
        self.num_categories = num_categories

    def ops_dict(self, **kwargs):
        x = tf.placeholder(tf.float32, shape=[None, self.window, ])
        y = tf.placeholder(tf.int32, shape=[None, ])

        encoder = functools.partial(voice_encoder, **kwargs)
        rnn = functools.partial(gru_rnn, **kwargs)
        classifier = functools.partial(linear_classifier, **kwargs)

        with tf.variable_scope("enc", reuse=False):
            z = encoder(tf.expand_dims(x, axis=-1), self.conv_dim)

        with tf.variable_scope("ar", reuse=False):
            context = rnn(z, self.ar_dim)

        with tf.variable_scope("classifier", reuse=False):
            context = tf.reduce_mean(context, axis=1)
            logits = classifier(context, self.num_categories)
        
        loss = None
        if self.num_categories == 1:
            loss = binary_ce_loss(y, logits)
        else:    
            loss = softmax_ce_loss(y, logits, self.num_categories)

        lr = tf.Variable(self.init_lr, trainable=False)
        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        train_op = optimizer.minimize(loss=loss)

        pred_binary = tf.nn.sigmoid(logits)
        pred_softmax = tf.nn.softmax(logits, axis=1)

        return EasyDict(
            x=x, y=y, z=z, 
            pred_binary=tf.reshape(pred_binary, [-1]), pred_softmax=pred_softmax,
            loss=loss, lr=lr, train_op=train_op)
