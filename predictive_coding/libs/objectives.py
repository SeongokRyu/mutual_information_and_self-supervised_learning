import tensorflow as tf

def info_nce(w_obj, z, context, future_step, num_future_steps=12):
    z_future = z[:, future_step:-(num_future_steps-future_step+1), :] # [B, T, F2]
    c_present = context[:, :-num_future_steps-1, :]  # [B, T, F1]
    c_present = tf.matmul(c_present, w_obj)

    # Shape of z_future : [bs, num_t, num_f]
    shape = tf.shape(z_future)
    bs = shape[0]
    num_t = shape[1]
    dim_z = shape[2]
    dim_c = tf.shape(c_present)[2]

    z_p = tf.transpose(z_future, [0,2,1])
    c_p = tf.transpose(c_present, [0,2,1])

    z_n = tf.reshape(z_p, [-1, num_t])
    c_n = tf.reshape(c_p, [-1, num_t])

    u_p = tf.matmul(z_p, c_present)
    u_p = tf.expand_dims(u_p, axis=2)
    u_n = tf.matmul(c_n, z_n, transpose_b=True)
    u_n = tf.reshape(u_n, [bs, dim_c, bs, dim_z])
    #u_n = tf.transpose(u_n, [0,2,3,1])

    pred_lgt = tf.concat([u_p, u_n], axis=2)
    pred_log = tf.nn.log_softmax(pred_lgt, axis=2)
    obj = tf.reduce_mean(pred_log[:,:,0])
    return tf.expand_dims(obj, axis=-1)

def binary_ce_loss(labels, logits):
    labels = tf.cast(labels, tf.float32)
    logits = tf.reshape(logits, [-1])
    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
    return tf.reduce_mean(loss)
    
def softmax_ce_loss(labels, logits, num_categories):
    labels = tf.one_hot(labels, num_categories)
    loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
    return tf.reduce_mean(loss)
