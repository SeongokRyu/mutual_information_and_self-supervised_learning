import tensorflow as tf
from easydict import EasyDict

class MINE:
    def __init__(self, hidden_dim, lr):
        self.hidden_dim = hidden_dim
        self.lr = lr

    def statistics_network(self):

        def two_layer_mlp(z, dim, reuse, act=tf.nn.sigmoid):
            out = tf.layers.dense(z, 
                                  units=dim, 
                                  activation=act,
                                  kernel_initializer=tf.glorot_normal_initializer(),
                                  name='linear1',
                                  reuse=reuse)
            out = tf.layers.dense(out, 
                                  units=1, 
                                  activation=None,
                                  kernel_initializer=tf.glorot_normal_initializer(),
                                  name='linear3',
                                  reuse=reuse)
            return out
    
        def kl_divergence(joint_score, marginals_score, eps=1e-5):
            mi = tf.reduce_mean(joint_score)
            mi -= tf.math.log(tf.reduce_mean(tf.math.exp(marginals_score))+eps)
            return mi

        x_inp = tf.placeholder(tf.float32, shape=[None,])
        y_inp = tf.placeholder(tf.float32, shape=[None,])

        batch_size = tf.shape(y_inp)[0]
        indices = tf.random.shuffle(tf.range(batch_size))
        y_shuffle = tf.gather(y_inp, indices)

        x = tf.expand_dims(x_inp, -1)
        y = tf.expand_dims(y_inp, -1)
        y_shuffle = tf.expand_dims(y_shuffle, -1)

        # Joint distribution and (the concatenation of ) its marginals.
        joint = tf.concat([x,y], axis=1)
        marginals = tf.concat([x, y_shuffle], axis=1)

        # A family of statistics networks as a two-layer MLP.
        with tf.variable_scope('statistics_network'):
            joint_score = two_layer_mlp(joint, self.hidden_dim, reuse=False)
            marginals_score = two_layer_mlp(marginals, self.hidden_dim, reuse=True) 
    
        mi_hat = kl_divergence(joint_score, marginals_score)
        train_op = tf.train.AdamOptimizer(self.lr).minimize(-mi_hat)
        return EasyDict(x=x_inp, y=y_inp, mi=mi_hat, train_op=train_op)
