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

import math
from struct import pack
from dissect import cstruct 
from .k210_constants import *
from tensorlayer import logging

#ummm, we implement one output first
def get_output_cnt(elayers):
    return 1

#kmodel v3 struct
#head: 
#   version,flag,arch,layers_length,
#   max_start_address,output_count,max_mem_usage
#outputs info:
#   address, size
#layer_headers:
#   type, body_size
#layers_bin:
#   xxx
#some args need 128byte align

def gen_kmodel(elayers, version, quant_bit):
    logging.info(" ")
    logging.info(" ")
    logging.info("=============================start gen_kmodel===================================")
    if version != 3:
        raise ValueError('only support kmodel v3 now!') 
        
    eight_bit_mode = (quant_bit==8)
    layer_number = len(elayers)
    output_number = get_output_cnt(elayers)
    
    # load kmodel structure defination (from C header)
    cparser = cstruct.cstruct()
    cparser.load(kmodel_def)
    header = cparser.kpu_model_header_t()
    outputs = [
        cparser.kpu_model_output_t()
        for i in range(output_number)                   
    ]
    layer_headers = [                       #fill header then
        cparser.kpu_model_layer_header_t()
        for i in range(layer_number)
	]
    layers_arg  = [0]*layer_number
    layers_bin  = [0]*layer_number
    layers_misc = [0]*layer_number
    memsize     = [0]*layer_number
    # struct kmodel header
    header.version          = 3
    header.flags            = eight_bit_mode           
    header.arch             = 0
    header.layers_length    = layer_number
    header.max_start_address= 0             #unknow
    header.output_count     = 1
    max_mem_usage    = 0             #update after all layer calculate

    header_length = len(header.dumps())
    outputs_length = len(outputs)*len(outputs[0].dumps())
    layer_headers_length = len(layer_headers)*len(layer_headers[0].dumps())
    
    #generate layers args
    arg_oft = header_length + outputs_length + layer_headers_length
    for idx in range(layer_number):
        elayer = elayers[idx]
        memsize[idx] = elayer.memsize
        if memsize[idx] > max_mem_usage:  #TODO: not support branch
            max_mem_usage = memsize[idx]
    buf_map = [max_mem_usage, 0, 0]
            
    for idx in range(layer_number):
        elayer = elayers[idx]
        logging.debug(" ")
        logging.debug("generate layer %d: %s"%(idx, elayer.typename))
        logging.debug("buf_map={}".format(buf_map))
        layer_headers[idx], layers_bin[idx], buf_map, layers_misc[idx]= \
            elayer.to_kmodel(arg_oft, eight_bit_mode, buf_map)
        #logging.debug(layers_bin[idx])
        arg_oft += len(layers_bin[idx])

    
    #fill output info
    for i in range(output_number):
        outputs[i].address = buf_map[2]              #oft to main_buffer
        outputs[i].size = elayers[layer_number-1].outsize
    #update max mem usage
    logging.debug("max_mem_usage=%d Byte, %d KB"%(max_mem_usage, max_mem_usage//1024))
    header.main_mem_usage = max_mem_usage   #TODO

    #combine the kmodel bin
    model_bin = header.dumps()
    
    for i in range(len(outputs)):
        model_bin += outputs[i].dumps()
        
    for i in range(len(layer_headers)):
        model_bin += layer_headers[i].dumps()
    
    for i in range(len(layer_headers)):
        model_bin += layers_bin[i]
        
    logging.info("===============================end gen_kmodel===================================")
    return model_bin
    

        
