import tensorflow as tf

def info_nce(w_obj, z, context, future_step, num_future_steps=12):
    z_future = z[:, future_step:-(num_future_steps-future_step+1), :] # [B, T, F2]
    c_present = context[:, :-num_future_steps-1, :]  # [B, T, F1]

    # For a joint distribution on local and global representations
    joint = tf.matmul(c_present, w_obj)
    joint = tf.reduce_sum(tf.multiply(joint, z_future), axis=-1)
    joint = tf.reduce_mean(joint)
    
    # For marginal distributions of the above joint distribution
    bs = tf.shape(z_future)[0]
    indices = tf.random.shuffle(tf.range(bs))

    c_n = tf.expand_dims(c_present, axis=1) 
    c_n = tf.tile(c_n, [1, bs, 1, 1])
    z_n = tf.expand_dims(z_future, axis=1)
    z_n = tf.tile(z_n, [1, bs, 1, 1])
    z_n = tf.transpose(tf.gather(z_n, indices), [1,0,2,3])

    c_n = tf.matmul(c_n, w_obj)
    marginal = tf.reduce_sum(tf.multiply(c_n, z_n), axis=3)
    #marginal = tf.log(tf.reduce_sum(tf.exp(marginal), axis=1)+1e-5)
    marginal = tf.reduce_logsumexp(marginal, axis=1)
    marginal = tf.reduce_mean(marginal)

    del z_future
    del c_present
    del c_n
    del z_n

    return tf.expand_dims(joint-marginal, axis=-1)

def binary_ce_loss(labels, logits):
    labels = tf.cast(labels, tf.float32)
    logits = tf.reshape(logits, [-1])
    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
    return tf.reduce_mean(loss)
    
def softmax_ce_loss(labels, logits, num_categories):
    labels = tf.one_hot(labels, num_categories)
    loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
    return tf.reduce_mean(loss)
