#! /usr/bin/python
# -*- coding: utf-8 -*-
"""MobileNet for ImageNet."""

import os

import tensorflow as tf
from tensorlayer import logging
from tensorlayer.files import (assign_weights, load_npz, maybe_download_and_extract)
from tensorlayer.layers import (ZeroPad2d, BatchNorm, Conv2d, DepthwiseConv2d, Flatten, GlobalMeanPool2d, Input, Reshape)
from tensorlayer.models import Model

__all__ = [
    'MobileNetV1',
]

layer_names = [
    'conv', 'depth1', 'depth2', 'depth3', 'depth4', 'depth5', 'depth6', 'depth7', 'depth8', 'depth9', 'depth10',
    'depth11', 'depth12', 'depth13', 'globalmeanpool', 'reshape', 'out'
]
n_filters = [32, 64, 128, 128, 256, 256, 512, 512, 512, 512, 512, 512, 1024, 1024]


def conv_block(n, n_filter, alpha, filter_size=(3, 3), strides=(1, 1), name='conv_block'):
    # ref: https://github.com/keras-team/keras/blob/master/keras/applications/mobilenet.py
    if strides != (1, 1):
        n = ZeroPad2d(padding=((1, 1), (1, 1)), name=name + '.pad')(n)
        padding_type = 'VALID'
    else:
        padding_type = 'SAME'
    n_filter = int(n_filter*alpha)
    n = Conv2d(n_filter, filter_size, strides, padding=padding_type, b_init=None, name=name + '.conv')(n)
    n = BatchNorm(decay=0.99, act=tf.nn.relu6, name=name + '.batchnorm')(n)
    return n


def depthwise_conv_block(n, n_filter, alpha, strides=(1, 1), name="depth_block"):
    if strides != (1, 1):
        n = ZeroPad2d(padding=((1, 1), (1, 1)), name=name + '.pad')(n)
        padding_type = 'VALID'
    else:
        padding_type = 'SAME'
    n_filter = int(n_filter*alpha)
    n = DepthwiseConv2d((3, 3), strides, padding=padding_type, b_init=None, name=name + '.depthwise')(n)
    n = BatchNorm(decay=0.99, act=tf.nn.relu6, name=name + '.batchnorm1')(n)
    n = Conv2d(n_filter, (1, 1), (1, 1), b_init=None, name=name + '.conv')(n)
    n = BatchNorm(decay=0.99, act=tf.nn.relu6, name=name + '.batchnorm2')(n)
    return n


def restore_params(network, alpha, path='models'):
    logging.info("Restore pre-trained parameters")
    #maybe_download_and_extract(
    #    'mobilenet.npz', path, 'https://github.com/tensorlayer/pretrained-models/raw/master/models/',
    #    expected_bytes=25600116
    #)  # ls -al
    filename = "mbnetv1_"+str(alpha)+".npz"
    params = load_npz(name=os.path.join(path, filename))  
    
    for idx, net_weight in enumerate(network.all_weights):
        if 'batchnorm' in net_weight.name:
            params[idx] = params[idx].reshape(1, 1, 1, -1)
    # exchange batchnorm's beta and gmma (TL and keras is different)
    idx = 0
    while idx < len(network.all_weights):
        net_weight = network.all_weights[idx]
        if ('batchnorm' in net_weight.name) and ('beta' in net_weight.name):
            tmp = params[idx]
            params[idx] = params[idx+1]
            params[idx+1] = tmp
            idx += 2
        else:
            idx += 1
            
    assign_weights(params[:len(network.all_weights)], network)
    del params


def MobileNetV1(pretrained=False, end_with='out', name=None, alpha=0.75):
    """Pre-trained MobileNetV1 model (static mode). Input shape [?, 224, 224, 3], value range [0, 1].

    Parameters
    ----------
    pretrained : boolean
        Whether to load pretrained weights. Default False.
    end_with : str
        The end point of the model [conv, depth1, depth2 ... depth13, globalmeanpool, out]. Default ``out`` i.e. the whole model.
    name : None or str
        Name for this model.

    Examples
    ---------
    Classify ImageNet classes, see `tutorial_models_mobilenetv1.py <https://github.com/tensorlayer/tensorlayer/blob/master/example/tutorial_models_mobilenetv1.py>`__

    >>> # get the whole model with pretrained weights
    >>> mobilenetv1 = tl.models.MobileNetV1(pretrained=True)
    >>> # use for inferencing
    >>> output = mobilenetv1(img1, is_train=False)
    >>> prob = tf.nn.softmax(output)[0].numpy()

    Extract features and Train a classifier with 100 classes

    >>> # get model without the last layer
    >>> cnn = tl.models.MobileNetV1(pretrained=True, end_with='reshape').as_layer()
    >>> # add one more layer and build new model
    >>> ni = Input([None, 224, 224, 3], name="inputs")
    >>> nn = cnn(ni)
    >>> nn = Conv2d(100, (1, 1), (1, 1), name='out')(nn)
    >>> nn = Flatten(name='flatten')(nn)
    >>> model = tl.models.Model(inputs=ni, outputs=nn)
    >>> # train your own classifier (only update the last layer)
    >>> train_params = model.get_layer('out').trainable_weights

    Returns
    -------
        static MobileNetV1.
    """
    ni = Input([None, 224, 224, 3], name="input")
    if alpha not in [0.25, 0.50, 0.75, 1.0]:
        raise ValueError('If imagenet weights are being loaded, '
                         'alpha can be one of'
                         '`0.25`, `0.50`, `0.75` or `1.0` only.')

    for i in range(len(layer_names)):
        if i == 0:
            n = conv_block(ni, n_filters[i], alpha, strides=(2, 2), name=layer_names[i])
        elif layer_names[i] in ['depth2', 'depth4', 'depth6', 'depth12']:
            n = depthwise_conv_block(n, n_filters[i], alpha, strides=(2, 2), name=layer_names[i])
        elif layer_names[i] == 'globalmeanpool':
            n = GlobalMeanPool2d(name='globalmeanpool')(n)
        elif layer_names[i] == 'reshape':
            n = Reshape([-1, 1, 1, int(1024*alpha)], name='reshape')(n)
        elif layer_names[i] == 'out':
            n = Conv2d(1000, (1, 1), (1, 1), name='out')(n)
            n = Flatten(name='flatten')(n)
        else:
            n = depthwise_conv_block(n, n_filters[i], alpha, name=layer_names[i])

        if layer_names[i] == end_with:
            break

    network = Model(inputs=ni, outputs=n, name=name)

    if pretrained:
        restore_params(network, alpha)

    return network
