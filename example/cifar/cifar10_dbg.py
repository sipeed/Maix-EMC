import tensorflow as tf
import tensorlayer as tl
import pprint
import numpy as np
from tensorlayer.models import Model
from tensorlayer.files import assign_weights
from tensorlayer.layers import (BatchNorm, Conv2d, Dense, Flatten, Input, LocalResponseNorm, MaxPool2d)

def restore_weight(net, M, layername):
    all_weights = net.all_weights
    for i in range(len(all_weights)-1):
        weights = all_weights[i]
        weights1 = all_weights[i+1]
        logging.debug(weights.name)
        if (layername in weights.name) and (layername not in weights1.name):
            break
    logging.debug(i)
    assign_weights(all_weights[0:i+1], M)

ni = Input([None, 32, 32, 3])
nn = Conv2d(32, (3, 3), (1, 1), padding='SAME', name='conv1')(ni)
nn = BatchNorm(decay=0.99, act=tf.nn.relu, name='batch1')(nn)
nn = MaxPool2d((3, 3), (2, 2), padding='SAME', name='pool1')(nn)
    
nn = Conv2d(64, (3, 3), (1, 1), padding='SAME', name='conv2')(nn)
nn = BatchNorm(decay=0.99, act=tf.nn.relu, name='batch2')(nn)
nn = MaxPool2d((3, 3), (2, 2), padding='SAME', name='pool2')(nn)

nn = Flatten(name='flatten')(nn)
nn = Dense(16, act=tf.nn.relu, name='denserelu')(nn)
nn = Dense(10, act=None, name='output')(nn)

M = Model(inputs=ni, outputs=nn, name='cnn1')

net = Model.load('./cifar10.h5', load_weights=True)
restore_weight(net, M, 'output')

filename='./cifar10_output'
M.save(filename+'.h5', save_weights=True)
M.save_k210(filename+'.kmodel', './data/cifar10_quant', dataset_func = 'img_0_1', quant_func='minmax', quant_bit=8, version=3)
