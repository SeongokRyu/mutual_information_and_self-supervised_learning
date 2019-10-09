import os
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from mine import MINE

#os.environ["CUDA_VISIBLE_DEVICES"] = "3"
flags = tf.app.flags
FLAGS = flags.FLAGS

def train(ops_dict):

    def generate_data(n_samples, eps):
        x = np.random.uniform(-3.0, 3.0, n_samples)
        noise = np.random.normal(0.0, 1.0, n_samples) * eps
        y = None
        if FLAGS.function=='linear':
            y = x**1
        elif FLAGS.function=='cubic':
            y = x**3
        elif FLAGS.function=='sin':
            y = np.sin(x)
        y += noise
        return x, y

    print ("Start training")
    x, y = generate_data(FLAGS.n_train, FLAGS.noise)
    n_batches = x.shape[0] // FLAGS.batch_size
    if x.shape[0]%FLAGS.batch_size != 0:
        n_batches+=1
    plt.figure()
    plt.title(FLAGS.function+'_'+str(FLAGS.noise)
    plt.scatter(x[0:100],y[0:100])
    plt.xlabel('x')
    plt.xlabel('y')
    plt.savefig(FLAGS.function+'_'+str(FLAGS.noise)+'_inputs.png')

    x_test, y_test = generate_data(FLAGS.n_train, FLAGS.noise)
    mi_list = []
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(FLAGS.n_epoches):
            xy = list(zip(x, y))
            random.shuffle(xy)
            x, y = [[x for x, y in xy],
                    [y for x, y in xy]]

            # Training
            for i in range(n_batches):
                xi = x[i*FLAGS.batch_size:(i+1)*FLAGS.batch_size]
                yi = y[i*FLAGS.batch_size:(i+1)*FLAGS.batch_size]

                feed_dict={ops_dict.x:xi, ops_dict.y:yi}
                _, mi = sess.run([ops_dict.train_op, ops_dict.mi], 
                                 feed_dict=feed_dict)
                print (epoch, "-th epoch", i, "-th batch, \t current MI=", mi)                
            print ("Finish training")

            # Test                
            mi_mean = 0.0
            num = 0
            for i in range(n_batches):
                num += 1
                xi = x_test[i*FLAGS.batch_size:(i+1)*FLAGS.batch_size]
                yi = y_test[i*FLAGS.batch_size:(i+1)*FLAGS.batch_size]

                feed_dict={ops_dict.x:xi, ops_dict.y:yi}
                mi = sess.run(ops_dict.mi, feed_dict=feed_dict)
                mi_mean += mi

            mi_mean /= num
            mi_list.append(mi_mean)
            print ("Average MI=", mi_mean, \
                   "at", epoch, "-th epoch")
            print ("Finish test")

    plt.figure()
    x = list(range(len(mi_list))
    plt.plot(x,mi_list)
    plt.xlabel('Epoch')
    plt.ylabel('Estimated MI')
    plt.savefig(FLAGS.function+'_'+str(FLAGS.noise)+'_mi.png')
    return

def main():
    model = MINE(FLAGS.hidden_dim, FLAGS.lr)
    ops_dict = model.statistics_network()
    train(ops_dict)
    return

if __name__ == '__main__':
    flags.DEFINE_integer('n_train', 2000, 'Number of training samples')
    flags.DEFINE_integer('n_test', 200, 'Number of test samples')
    flags.DEFINE_float('noise', 0.1, 'Amount of noise level')
    flags.DEFINE_integer('hidden_dim', 64, 'Hidden dimension of MLP networks')
    flags.DEFINE_integer('batch_size', 200, 'Batch size')
    flags.DEFINE_integer('n_epoches', 200, 'Number of training epoches')
    flags.DEFINE_float('lr', 5e-4, 'Learning rate')
    flags.DEFINE_float('ema', 0.999, 'Decay rate of exponential moving average (ema)')
    flags.DEFINE_string('function', 'cubic', 'Options: linear, cubic, sin')
    main()
