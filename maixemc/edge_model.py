#! /usr/bin/python
# -*- coding: utf-8 -*-

'''
 * Copyright 2019 Sipeed Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 '''

import base64
import gzip
import json
import math
import os
import pickle
import re
import shutil
# import ast
import sys
import tarfile
import time
import zipfile

import h5py
import numpy as np
import scipy.io as sio
from six.moves import cPickle

import progressbar
import tensorflow as tf
import tensorlayer as tl
from tensorflow.python.keras.saving import model_config as model_config_lib
from tensorflow.python.platform import gfile
from tensorlayer import logging, nlp, utils, visualize

import cloudpickle
import base64
from tensorflow.python.util.tf_export import keras_export
from tensorflow.python.util import serialization
import json
import datetime
from tensorlayer import logging
from .edge_constants import * 
from .edge_layer import * 
from .edge_quant import quant_func_valid,quant_func_byname
from .edge_dataloader import loader_func_valid, loader_func_byname

from .k210.k210_layer import gen_k210_conv_layer, k210_layer_post_fix
from .k210.k210_kmodel_bin import gen_kmodel
   

tl_to_k210_table= {
#   TL layer class          layer_generator     merge
    'Dense'                 :[gen_fc_layer,         [[],]] ,
    'Flatten'               :[gen_flatten_layer,    [[],]] ,
    'Reshape'               :[None,                 [[],]] ,
    'GlobalMaxPool2d'       :[gen_gmaxpool2d_layer, [[],]] ,
    'GlobalMeanPool2d'      :[gen_gavgpool2d_layer, [[],]] ,
    'MaxPool2d'             :[gen_maxpool2d_layer,  [[],]] ,
    'MeanPool2d'            :[gen_avgpool2d_layer,  [[],]] ,
    'Concat'                :[gen_concat_layer,     [[],]] ,
    'Conv2d'                :[gen_k210_conv_layer,  [[], ['BatchNorm'],]] ,
    'DepthwiseConv2d'       :[gen_k210_conv_layer,  [[], ['BatchNorm'],]] ,
    'ZeroPad2d'             :[gen_k210_conv_layer,  [['Conv2d'], \
                                                     ['Conv2d', 'BatchNorm'], \
                                                     ['DepthwiseConv2d'], \
                                                     ['DepthwiseConv2d', 'BatchNorm']]] ,
    'DummyDequant'          :[gen_dequant_layer,    [[],]] ,  
    'SoftMax'               :[gen_softmax_layer,    [[],]] ,  
}



platform_table = {
#   platform       tl layer convertor   model generator     post_fix_func
    'k210'      :   [tl_to_k210_table, gen_kmodel,      k210_layer_post_fix]
    #'stm32'    :   gen_stm32_layer_func_table,
} 


def check_is_seq(layers, idx):
    last_layer_node = layers[idx-1]['args']['name']+'_node_0'
    return (layers[idx]['prev_layer'][0] == last_layer_node)


# get the longest match layer list
def try_match_layer_type(layers, idx, match_table):
    if layers[idx]['class'] in match_table:
        tl_type         = layers[idx]['class']
        gen_func        = match_table[tl_type][0]
        tl_type_lists   = match_table[tl_type][1]
        
        if len(tl_type_lists) == 0:
            return gen_func, [tl_type]
            
        list_result = []
        #try all layer list belong to this k210 layer type
        for tl_type_list in  tl_type_lists:    
            if len(tl_type_list) == 0:
                list_result.append(tl_type_list)
                continue
            match_flag = True
            for i in range(len(tl_type_list)):  #match tl layer type in list
                if check_is_seq(layers, idx+i) == False:
                    raise RuntimeError("only support seq structure yet !")
                if layers[idx+i+1]['class'] != tl_type_list[i]:
                    match_flag = False
                    break
            if match_flag:
                list_result.append(tl_type_list)
        if len(list_result) == 0:               #no match tl layer type list
            return None, []
            
        max_len_idx = 0
        max_len = 0
        for i in range(len(list_result)):
            if len(list_result[i]) > max_len:
                max_len = len(list_result[i])
                max_len_idx = i                 #return max len match result
        tl_type_list = [tl_type]
        if max_len != 0:
            tl_type_list.extend(list_result[max_len_idx])
        return gen_func, tl_type_list
    else:
        return None, []


# gen layer list logging.info
def gen_tl_type_list(layers, idx):
    type_list=[]
    for i in range(idx, len(layers)):
        layer_class = layers[i]['class']
        type_list.append(layer_class)
    return type_list
    
# gen single layer
def gen_edge_layer_from_network(network, platform, meta_info, idx):
    logging.info("==================start gen_edge_layers_from_network========================")
    pool_tensor_info = dict()
    pool_type_size_stride = None  # bypass pool
    layers = network.config['model_architecture']
    
    if platform not in platform_table:
        raise RuntimeError("not support this platform !")
    match_table = platform_table[platform][0]
        
    gen_layer_func, tl_type_list = try_match_layer_type(layers, idx, match_table)
    if len(tl_type_list) == 0:
        type_list = gen_tl_type_list(layers, idx)
        logging.error("[*] This layer sequence not support: {}".format(type_list))
        raise RuntimeError("not support this layer sequence !")
        
    logging.info("This layerlist len = %d"%(len(tl_type_list)))
    for i in range(len(tl_type_list)):
        logging.info("[TL layer %d]: %s"%(idx+i, network.all_layers[idx+i].layer_args['name']))
    
    if gen_layer_func != None:
        el_layers, meta_info = gen_layer_func(
            network,idx, tl_type_list, meta_info)
    else:   #dummy layer
        logging.info("[note] Dummy layer")
        el_layers = []
    
    logging.info("====================end gen_edge_layers_from_network========================")
    return el_layers, idx+len(tl_type_list), meta_info

# combine duplicate layers  
def post_fix_layers(el_list, platform_fix_func):
    logging.info(' ')
    logging.info(' ')
    logging.info('='*27+'Layer List before fix'+'='*27)
    for i in range(len(el_list)):
        logging.info("Layer %3d: %s"%(i,el_list[i].typename))
    
    if len(el_list) <= 1:
        return
    i = 0
    while i < len(el_list)-1:
        ty0 = el_list[i].type
        ty1 = el_list[i+1].type
        if (ty0 == EL_DEQUANTIZE and ty1 == EL_QUANTIZE) or \
           (ty0 == EL_QUANTIZE and ty1 == EL_DEQUANTIZE):
           el_list.remove(el_list[i])
           el_list.remove(el_list[i])
           continue
        i += 1
        
    platform_fix_func(el_list)

    logging.info('='*27+'Layer List after fix '+'='*27)
    for i in range(len(el_list)):
        logging.info("Layer %3d: %s"%(i,el_list[i].typename))
    logging.info('='*80)
    return el_list
    
#gen edge layers info
def gen_edge_layers_from_network(network, platform, dataset, quant_func, quant_bit=8, sm_flag=False):
    logging.info("====================start gen_edge_layers_from_network==========================")
    layers = network.config['model_architecture']
    layer_length = len(layers)
    el_list = []
    index = 1           #layer after input
    is_quant = True     #assume first layer is quant 
    assert (layers[0]['class']=='_InputLayer')
    
    input_layer = network.all_layers[0]
    last_min, last_max, *_ = quant_func(network, input_layer, dataset, is_weights=False)
    
    # First, we get all layers args,quant them
    conv_idx = 0
    is_inai = True  #is it in ai ram
    meta_info = { \
        'dataset'   : dataset, 
        'quant_bit' : quant_bit, 
        'quant_func': quant_func, 
        'last_min'  : last_min, 
        'last_max'  : last_max, 
        'is_quant'  : is_quant, 
        'conv_idx'  : conv_idx, 
        'is_inai'   : is_inai}
    while index < layer_length:
        logging.info("  ")
        logging.info("tl layer index = %d, total cnt=%d"%(
            index, layer_length))
        cur_edge_layer, index, meta_info = \
        gen_edge_layer_from_network(
            network, platform, meta_info, index
        )
        el_list.extend(cur_edge_layer)  
    # last layer add DEQUANTIZE
    if platform not in platform_table:
        raise RuntimeError("not support this platform !")
    match_table = platform_table[platform][0]
    gen_layer_func = match_table['DummyDequant'][0]
    el_layer, _ = gen_layer_func(network,layer_length-1, None, meta_info)
    el_list.extend(el_layer)  
    # add optional softmax layer
    if sm_flag:
        gen_layer_func = match_table['SoftMax'][0]
        el_layer, _ = gen_layer_func(network,layer_length-1, None, meta_info)
        el_list.extend(el_layer) 
    
    platform_fix_func = platform_table[platform][2]
    el_list = post_fix_layers(el_list, platform_fix_func)
    logging.info("======================end gen_edge_layers_from_network==========================")
    return el_list


###############################################################################
# public functions 

def available_platform():
    for item in platform_table:
        logging.info("######"+item)
        logging.info("support layer type:")
        table = platform_table[item][0]
        for layer in table:
            logging.info(layer)


# gen edge model from network
def gen_edge_model(network, platform, version, dataset_dir, dataset_func, quant_func, quant_bit, sm_flag):
    if platform not in platform_table:
        raise RuntimeError("unsupport platform!")
    
    if isinstance(quant_func,str):
        if quant_func_valid(quant_func) == False:
            raise RuntimeError("save kmodel not support your quant function now!")
        else:
            quant_func = quant_func_byname(quant_func)
    else:
        raise RuntimeError("only support str quant func name")
    if isinstance(dataset_func,str):
        if loader_func_valid(dataset_func) == False:
            raise RuntimeError("save kmodel not support your dataset loader function now!")
        else:
            dataset_loader_func = loader_func_byname(dataset_func)
    else:
        raise RuntimeError("only support str dataset loader func name")
    config = network.config
    layers_info=config['model_architecture']
    input_layer_info = layers_info[0]    #TODO: is first layer must index 0?
    # load dataset
    input_shape = input_layer_info['args']['shape']
    input_name = input_layer_info['args']['name']
    input_w = input_shape[1]
    input_h = input_shape[2]
    input_ch = input_shape[3]
    dataset_val = dataset_loader_func(dataset_dir, input_w, input_h, input_ch)
    # generate edge layers
    edge_layers = gen_edge_layers_from_network(
        network, platform, 
        dataset_val, quant_func, quant_bit, sm_flag
    )

    # generate model for edge device
    gen_model_func = platform_table[platform][1]
    output_model = gen_model_func(edge_layers, version, quant_bit)
    model_size = len(output_model)
    logging.info("kmodel size = %d Byte, %d KB"%(model_size, model_size//1024))
    return output_model
 
 
def save_kmodel(network, filepath, dataset_dir, dataset_func='img_0_1', quant_func='minmax', quant_bit=8, version=3, sm_flag=False):
    if network.outputs is None:
        raise RuntimeError("save_kmodel_graph not support dynamic mode yet")

    logging.info("[*] Saving TL model into {}, dataset path={}, quant func is {}, quant bit is {}, kmodel version is {}".format(
        filepath, dataset_dir, quant_func, quant_bit, version))

    try:
        output_kmodel = gen_edge_model(network, 'k210', version, dataset_dir, dataset_func, quant_func, quant_bit, sm_flag)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as of:
            of.write(output_kmodel)
        logging.info(
            "[*] Saved TL model into {}, dataset path={}, quant func is {}, quant bit is {}, kmodel version is {}".format(
            filepath, dataset_dir, quant_func, quant_bit, version))
    except Exception as e:
        raise RuntimeError("save kmodel error")
        