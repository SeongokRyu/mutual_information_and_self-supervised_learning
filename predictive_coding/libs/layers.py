import tensorflow as tf

def conv1d(x, filters, kernel_size, strides, padding='same', activation=tf.nn.relu):
    conv_args = dict(filters=filters,
                     kernel_size=kernel_size,
                     strides=strides,
                     padding=padding,
                     activation=activation,
                     kernel_initializer=tf.glorot_normal_initializer(),
                     bias_initializer=tf.zeros_initializer())
    return tf.layers.conv1d(x, **conv_args)

def dense(x, units, activation=None, use_bias=True):
    dense_args = dict(units=units,
                      activation=activation,
                      use_bias=use_bias,
                      kernel_initializer=tf.glorot_normal_initializer(),
                      bias_initializer=tf.zeros_initializer())
    return tf.layers.dense(x, **dense_args)   
