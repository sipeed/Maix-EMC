#! /usr/bin/python
# -*- coding: utf-8 -*-
"""MobileNet for ImageNet."""

import os
import maixemc as emc

import tensorflow as tf
import tensorlayer as tl
from tensorlayer import logging
from tensorlayer.files import (assign_weights, load_npz, maybe_download_and_extract)
from tensorlayer.layers import (ZeroPad2d, BatchNorm, Conv2d, DepthwiseConv2d, Flatten, GlobalMeanPool2d, Input, Reshape)
from tensorlayer.models import Model
import numpy as np


__all__ = [
    'MobileNetV1',
]

layer_names_all = [
    'conv', 'depth1', 'depth2', 'depth3', 'depth4', 'depth5', 'depth6', 'depth7', 'depth8', 'depth9', 'depth10',
    'depth11', 'depth12', 'depth13', 'globalmeanpool', 'reshape', 'out'
]
n_filters = [32, 64, 128, 128, 256, 256, 512, 512, 512, 512, 512, 512, 1024, 1024]


def conv_block(n, n_filter, filter_size=(3, 3), strides=(1, 1), name='conv_block'):
    # ref: https://github.com/keras-team/keras/blob/master/keras/applications/mobilenet.py
    if strides != (1, 1):
        n = ZeroPad2d(padding=((1, 1), (1, 1)), name=name + '.pad')(n)
        padding_type = 'VALID'
    else:
        padding_type = 'SAME'
    n = Conv2d(n_filter, filter_size, strides, padding=padding_type, b_init=None, name=name + '.conv')(n)
    n = BatchNorm(decay=0.99, act=tf.nn.relu6, name=name + '.batchnorm')(n)
    return n


def depthwise_conv_block(n, n_filter, strides=(1, 1), name="depth_block", full_flag = True):
    if strides != (1, 1):
        n = ZeroPad2d(padding=((1, 1), (1, 1)), name=name + '.pad')(n)
        padding_type = 'VALID'
    else:
        padding_type = 'SAME'
    n = DepthwiseConv2d((3, 3), strides, padding=padding_type, b_init=None, name=name + '.depthwise')(n)
    n = BatchNorm(decay=0.99, act=tf.nn.relu6, name=name + '.batchnorm1')(n)
    if full_flag:
        n = Conv2d(n_filter, (1, 1), (1, 1), b_init=None, name=name + '.conv')(n)
        n = BatchNorm(decay=0.99, act=tf.nn.relu6, name=name + '.batchnorm2')(n)
    return n


def restore_params(network, path='models'):
    logging.info("Restore pre-trained parameters")
    maybe_download_and_extract(
        'mobilenet.npz', path, 'https://github.com/tensorlayer/pretrained-models/raw/master/models/',
        expected_bytes=25600116
    )  # ls -al
    params = load_npz(name=os.path.join(path, 'mobilenet.npz'))
    for idx, net_weight in enumerate(network.all_weights):
        if 'batchnorm' in net_weight.name:
            params[idx] = params[idx].reshape(1, 1, 1, -1)
    assign_weights(params[:len(network.all_weights)], network)
    del params


def MobileNetV1(pretrained=False, end_with='out', name=None):
    ni = Input([None, 224, 224, 3], name="input")
    for i in range(len(layer_names)):
        if i == len(layer_names)-1:
            full_flag = full
        else:
            full_flag = True
        if i == 0:
            n = conv_block(ni, n_filters[i], strides=(2, 2), name=layer_names[i])
        elif layer_names[i] in ['depth2', 'depth4', 'depth6', 'depth12']:
            n = depthwise_conv_block(n, n_filters[i], strides=(2, 2), name=layer_names[i], full_flag = full_flag)
        elif layer_names[i] == 'globalmeanpool':
            n = GlobalMeanPool2d(name='globalmeanpool')(n)
        elif layer_names[i] == 'reshape':
            n = Reshape([-1, 1, 1, 1024], name='reshape')(n)
        elif layer_names[i] == 'out':
            n = Conv2d(1000, (1, 1), (1, 1), name='out')(n)
            n = Flatten(name='flatten')(n)
        else:
            n = depthwise_conv_block(n, n_filters[i], name=layer_names[i], full_flag = full_flag)

        if layer_names[i] == end_with:
            break

    network = Model(inputs=ni, outputs=n, name=name)

    if pretrained:
        restore_params(network)
    return network
    
#17 total
print("total %d layers"%len(layer_names_all))
tl.logging.set_verbosity(tl.logging.DEBUG)
dbg_layer_idx = 12 #len(layer_names_all)+1 #4
full = True

layer_names = layer_names_all[:dbg_layer_idx]
mobilenetv1 = MobileNetV1(pretrained=True)
emc.save_kmodel(mobilenetv1, './mbnetv1.kmodel', './mbnetv1_dataset', dataset_func='img_0_1', quant_func='minmax', quant_bit=8, version=3, sm_flag=True)
os.system('zip mbnetv1.kfpkg mbnetv1.kmodel flash-list.json')

np.set_printoptions(suppress=True)
np.set_printoptions(threshold=100000)

img1 = tl.vis.read_image('mbnetv1_dataset/black.bmp')
img1 = tl.prepro.imresize(img1, (224, 224)) / 255
img1 = img1.astype(np.float32)[np.newaxis, ...]
mobilenetv1.eval()
output = mobilenetv1(img1).numpy()
print(output.shape)
print(output)
#print(output[0][0][0])
#output = output.swapaxes(1,3)
#output = output.swapaxes(2,3)
#print(output[0][1])


img1 = tl.vis.read_image('mbnetv1_dataset/tiger224.bmp')
img1 = tl.prepro.imresize(img1, (224, 224)) / 255
img1 = img1.astype(np.float32)[np.newaxis, ...]
mobilenetv1.eval()
output = mobilenetv1(img1).numpy()
print(output.shape)
print(output)
#print(output[0][0][0])
#output = output.swapaxes(1,3)
#output = output.swapaxes(2,3)
#print(output[0][1])