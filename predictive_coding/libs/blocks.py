import tensorflow as tf
from libs.layers import dense
from libs.layers import conv1d

def voice_encoder(x, hidden_units=512):
    kernels = [10, 8, 4, 4, 4]
    strides = [5, 4, 2, 2, 2]
    h = x
    for i in range(len(strides)):
        h = conv1d(h, 
                   filters=hidden_units, 
                   kernel_size=kernels[i], 
                   strides=strides[i])
    return h

def gru_rnn(x, hidden_units=256):
    return tf.keras.layers.GRU(units=hidden_units, 
                               return_sequences=True)(x) 

def linear_classifier(x, num_categories):
    return dense(x, num_categories)    
