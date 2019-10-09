import tensorflow as tf

def spectral_normalization(weight, num_iter, idx):
    w_shape = weight.shape.as_list()
    weight = tf.reshape(weight, [-1, w_shape[-1]])
    u = tf.get_variable("u_sn"+str(idx), 
                        shape=[1, w_shape[-1]], 
                        initializer=tf.random_normal_initializer(), 
                        trainable=False)

    u_hat = u
    v_hat = None
    for i in range(num_iter):
        v_ = tf.matmul(u_hat, tf.transpose(weight))
        v_hat = tf.nn.l2_normalize(v_)

        u_ = tf.matmul(v_hat, weight)
        u_hat = tf.nn.l2_normalize(u_)

    u_hat = tf.stop_gradient(u_hat)
    v_hat = tf.stop_gradient(v_hat)
    sigma = tf.matmul(tf.matmul(v_hat, weight), tf.transpose(u_hat))

    with tf.control_dependencies([u.assign(u_hat)]):
        w_norm = weight / tf.maximum(1.0, sigma)
        w_norm = tf.reshape(w_norm, w_shape)
    return w_norm
