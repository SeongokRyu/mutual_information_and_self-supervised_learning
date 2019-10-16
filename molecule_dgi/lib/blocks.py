import tensorflow as tf
import numpy as np

def conv1d(x, out_dim, use_bias=True):
    output = tf.layers.conv1d(x,
                              filters=out_dim,
                              kernel_size=1,
                              activation=None,
                              use_bias=use_bias,
                              kernel_initializer=tf.glorot_normal_initializer(),
                              bias_initializer=tf.glorot_normal_initializer())
    return output    

def conv1d_with_dropout(x, out_dim, is_training, FLAGS, use_bias=True):
    output = conv1d(x, out_dim, use_bias)
    output = tf.layers.dropout(output, rate=FLAGS.dropout, training=is_training)
    return output    

def dense(x, out_dim, use_bias=True):
    output = tf.layers.dense(x,
                             units=out_dim,
                             use_bias=use_bias,
                             activation=None,
                             kernel_initializer=tf.glorot_normal_initializer(),
                             bias_initializer=tf.glorot_normal_initializer())
    return output    

def dense_with_dropout(x, out_dim, is_training, FLAGS):
    output = dense(x, out_dim)
    output = tf.layers.dropout(output, rate=FLAGS.dropout, training=is_training)
    return output    

def attn_matrix(adj, x, FLAGS):
    # A : [batch, N, N]
    # X : [batch, N, F']
    # weight_attn : F' 
    num_features = int(x.get_shape()[2])
    x1 = conv1d(x, num_features, use_bias=False)
    x1 = tf.transpose(x1, [0,2,1])
    x1 = tf.matmul(x, x1)
    adj = tf.multiply(adj, x1)
    adj /= np.sqrt(num_features)
    return tf.nn.tanh(adj)

def skip_connection(h, x):
    if( int(h.get_shape()[2]) != int(x.get_shape()[2]) ):
       out_dim = int(h.get_shape()[2])
       x = conv1d(x, out_dim, use_bias=False)
    return h+x

def graph_conv(adj, x, is_training, FLAGS):
    # Graph convolution w/o attention
    dim = FLAGS.hidden_dim

    h = conv1d(x, dim, use_bias=False)
    h = tf.matmul(adj, h)
    h = tf.nn.elu(h)
    h = tf.layers.dropout(h, rate=FLAGS.dropout, training=is_training)

    h = skip_connection(h, x) 
    if FLAGS.normalization:
        h = tf.contrib.layers.layer_norm(h)

    h2 = conv1d(h, 4*FLAGS.hidden_dim, True)
    h2 = tf.nn.elu(h2)
    h2 = conv1d(h2, FLAGS.hidden_dim, True)
    h2 = tf.layers.dropout(h2, rate=FLAGS.dropout, training=is_training)

    h = skip_connection(h2, h)
    if FLAGS.normalization:
        h = tf.contrib.layers.layer_norm(h)
    return h

def graph_conv_attn(adj, x, is_training, FLAGS):
    # Self-attention with K-multi heads
    h_total = []
    for i in range(FLAGS.num_attn):
        dim = (FLAGS.hidden_dim//FLAGS.num_attn)
        h = conv1d(x, dim, use_bias=False)
        attn = attn_matrix(adj, h, FLAGS)
        h = tf.matmul(attn, h)
        h_total.append(h)
    h_total = tf.concat(h_total, 2)
    h = conv1d(h_total, FLAGS.hidden_dim, False)
    h = tf.nn.elu(h)
    h = tf.layers.dropout(h, rate=FLAGS.dropout, training=is_training)

    h = skip_connection(h, x) 
    if FLAGS.normalization:
        h = tf.contrib.layers.layer_norm(h)

    h2 = conv1d(h, 4*FLAGS.hidden_dim, True)
    h2 = tf.nn.elu(h2)
    h2 = conv1d(h2, FLAGS.hidden_dim, True)
    h2 = tf.layers.dropout(h2, rate=FLAGS.dropout, training=is_training)

    h = skip_connection(h2, h)
    if FLAGS.normalization:
        h = tf.contrib.layers.layer_norm(h)
    return h

def encoder_gcn_attn(adj, x, is_training, FLAGS):
    # X : Atomic Feature, A : Adjacency Matrix
    h = x
    for i in range(FLAGS.num_layers):
        h = graph_conv_attn(adj, h, is_training, FLAGS)
    return h

def encoder_gcn(adj, x, is_training, FLAGS):
    # X : Atomic Feature, A : Adjacency Matrix
    h = x
    for i in range(FLAGS.num_layers):
        h = graph_conv(adj, h, is_training, FLAGS)
    return h

def readout_atomwise(x, FLAGS):
    z = conv1d(x, FLAGS.latent_dim)
    z = tf.nn.sigmoid(tf.reduce_mean(z, 1))
    return z

def readout_graph_gather(x, FLAGS):
    z = tf.multiply(tf.nn.sigmoid(conv1d(x, FLAGS.latent_dim)), 
                                  conv1d(x, FLAGS.latent_dim))
    z = tf.nn.sigmoid(tf.reduce_mean(z, 1))
    return z

def linear_predictor(z):
    # Predict the molecular property
    return dense(z, 2) # no dropout
