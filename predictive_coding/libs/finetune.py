import re
import collections
import functools
import tensorflow as tf

from easydict import EasyDict

from libs.blocks import voice_encoder
from libs.blocks import gru_rnn
from libs.blocks import linear_classifier
from libs.objectives import binary_ce_loss
from libs.objectives import softmax_ce_loss

class Finetune():

    def __init__(self,
                 pre_trained,
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
        self.pre_trained = pre_trained

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

        var_total = tf.trainable_variables()
        var_reuse = []
        var_classifier = []
        for v in var_total:
            if 'enc' in v.name:
                var_reuse.append(v)

            if 'ar' in v.name:
                var_reuse.append(v)
            
            if 'classifier' in v.name:
                var_classifier.append(v)    

        init_checkpoint = './save/'+self.pre_trained
        assignment_map, initialized_variable_names = self.get_assignment_map_from_checkpoint(
            var_reuse, init_checkpoint)
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        lr = tf.Variable(self.init_lr, trainable=False)
        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        train_op = optimizer.minimize(loss=loss, var_list=var_classifier)

        pred_binary = tf.nn.sigmoid(logits)
        pred_softmax = tf.nn.softmax(logits, axis=1)

        return EasyDict(
            x=x, y=y, z=z, 
            pred_binary=tf.reshape(pred_binary, [-1]), pred_softmax=pred_softmax,
            loss=loss, lr=lr, train_op=train_op)

    def get_assignment_map_from_checkpoint(self, trainable_variables, init_checkpoint):
        assignment_map = {}
        initialized_variable_names = {}

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
            #assignment_map[name] = name
            assignment_map[name] = name_to_variable[name]
            initialized_variable_names[name] = 1
            initialized_variable_names[name+":0"] = 1
        return assignment_map, initialized_variable_names
