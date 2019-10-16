import tensorflow as tf

def sharpening(y, temperature):
    y0 = y**(1.0/temperature)
    y1 = (1.0-y)**(1.0/temperature)
    output = y0/(y0+y1)
    return output

def brier_score(y_truth, y_pred):
    y_pred = tf.reshape(y_pred, [-1])
    y_pred = tf.nn.sigmoid(y_pred)
    output = tf.reduce_mean((y_truth - y_pred)**2)
    return output    

def bce_loss(y_truth, y_pred):
    y_truth = tf.reshape(y_truth, [-1])
    y_pred = tf.reshape(y_pred, [-1])

    nll = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_truth, logits=y_pred))
    reg_loss = tf.reduce_sum(tf.losses.get_regularization_losses())
    total_loss = nll + reg_loss
    return total_loss, nll

def heteroscedastic_l2(y_truth, y_mean, y_logvar):
    y_truth = tf.reshape(y_truth, [-1])
    y_mean = tf.reshape(y_mean, [-1])
    y_logvar = tf.reshape(y_logvar, [-1])

    nll = tf.reduce_mean(tf.exp(-y_logvar)*(y_truth - y_mean)**2 + y_logvar)
    reg_loss = tf.reduce_sum(tf.losses.get_regularization_losses())
    total_loss = nll + reg_loss
    return total_loss, nll    

def loss_attenuation(y_truth, y_mean, y_logvar):
    y_truth = tf.reshape(y_truth, [-1])
    y_mean = tf.reshape(y_mean, [-1])
    y_logvar = tf.reshape(y_logvar, [-1])
    
    eps = tf.random_normal(shape=tf.shape(y_mean), mean=0.0, stddev=1.0)
    y_pred = y_mean + y_logvar * eps

    nll = -tf.reduce_mean(y_truth*y_pred - tf.log(1.0 + tf.exp(y_pred)))
    reg_loss = tf.reduce_sum(tf.losses.get_regularization_losses())
    total_loss = nll + reg_loss
    return total_loss, nll

def mi_js_divergence(node_feature, graph_feature, kernel_d):        
    batch_size = tf.shape(node_feature)[0]
    n_atoms = tf.shape(node_feature)[1]

    graph_copied = tf.expand_dims(graph_feature, axis=1)
    graph_copied = tf.tile(graph_copied, [1,n_atoms,1])

    real_score = tf.matmul(node_feature, kernel_d)
    real_score = tf.reduce_sum(
        tf.multiply(real_score, graph_copied), axis=2)

    indices = tf.random.shuffle(tf.range(batch_size))
    node_feature_shuffle = tf.gather(node_feature, indices)
    fake_score = tf.matmul(node_feature_shuffle, kernel_d)
    fake_score = tf.reduce_sum(
        tf.multiply(fake_score, graph_copied), axis=2)

    fake_score = tf.nn.sigmoid(fake_score)
    real_score = tf.nn.sigmoid(real_score)
    eps = 1e-5

    real_score = tf.reduce_mean(tf.log(real_score+eps))
    fake_score = tf.reduce_mean(tf.log(1.0-fake_score+eps))

    loss = -0.5*(real_score + fake_score)
    return loss

def mi_kl_divergence(node_feature, graph_feature, kernel_d):        
    batch_size = tf.shape(node_feature)[0]
    n_atoms = tf.shape(node_feature)[1]

    graph_copied = tf.expand_dims(graph_feature, axis=1)
    graph_copied = tf.tile(graph_copied, [1,n_atoms,1])

    real_score = tf.matmul(node_feature, kernel_d)
    real_score = tf.reduce_sum(
        tf.multiply(real_score, graph_copied), axis=2)

    indices = tf.random.shuffle(tf.range(batch_size))
    node_feature_shuffle = tf.gather(node_feature, indices)
    fake_score = tf.matmul(node_feature_shuffle, kernel_d)
    fake_score = tf.reduce_sum(
        tf.multiply(fake_score, graph_copied), axis=2)

    vectors = tf.concat([real_vectors, fake_vectors], axis=1)
    vector_shape = tf.shape(real_vectors)
    labels = tf.concat(
        [tf.ones(shape=vector_shape), tf.zeros(shape=vector_shape)],
        axis=1)
    
    loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=vectors))
    return loss

def mi_info_nce(node_feature, graph_feature, kernel_d):        
    batch_size = tf.shape(node_feature)[0]
    n_atoms = tf.shape(node_feature)[1]

    # For positive samples (real score)
    graph_copied = tf.expand_dims(graph_feature, axis=1)
    graph_copied = tf.tile(graph_copied, [1,n_atoms,1])

    real_score = tf.matmul(node_feature, kernel_d)
    real_score = tf.reduce_sum(
        tf.multiply(real_score, graph_copied), axis=2)
    real_score = tf.reduce_mean(real_score)

    # For negative samples (fake score)
    indices = tf.random.shuffle(tf.range(batch_size))

    graph_copied = tf.expand_dims(graph_copied, axis=1)
    graph_copied = tf.tile(graph_copied, [1, batch_size, 1, 1])
    node_shuffle = tf.expand_dims(node_feature, axis=1)
    node_shuffle = tf.tile(node_shuffle, [1, batch_size, 1, 1])
    node_shuffle = tf.transpose(tf.gather(node_shuffle, indices), [1,0,2,3])

    fake_score = tf.matmul(node_shuffle, kernel_d)
    fake_score = tf.reduce_sum(tf.multiply(graph_copied, fake_score), axis=3)
    fake_score = tf.reduce_logsumexp(fake_score, axis=1)
    fake_score = tf.reduce_mean(fake_score)

    del graph_copied
    del node_shuffle
    return fake_score-real_score
