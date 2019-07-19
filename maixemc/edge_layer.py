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

from .edge_constants import * 
from .k210.k210_constants import *
from struct import pack
from dissect import cstruct 
import struct
from tensorlayer import logging

#    utils
def round_size_4(n):
    return (n+4-1)//4*4
def round_size_8(n):
    return (n+8-1)//8*8
def round_size_128(n):
    return (n+128-1)//128*128
    
def min_max_to_scale_bias(minv, maxv):
    scale = (maxv - minv) / 255
    bias = minv
    return scale, bias  
    
def get_tensor_size(shape):
    if len(shape) < 2:
        raise RuntimeError("in/out tensor must > 2D!")
    tsize = 1
    for i in range(len(shape)-1):
        tsize = tsize * shape[1+i]
    return tsize
    
def flaot2bytes(f):
    bs = struct.pack("<f",f)
    return bs

#buf_map,:[buf_size, pingpong_flag, last_addr]
def cal_in_out_addr(buf_map, outsize):
    bufsize         = buf_map['bufsize']
    pingpong_flag   = buf_map['pingpong']
    last_addr       = buf_map['last_addr']
    if outsize == 0:
        logging.info("outsize==0, no need buf")
        return buf_map,last_addr,last_addr
        
    in_addr = last_addr
    out_addr = (0 if last_addr!=0 else bufsize-outsize)
    
    buf_map['pingpong'] = 1 - buf_map['pingpong']
    buf_map['last_addr'] = out_addr
    logging.info("buf_size=%x, outsize=%x, in_addr=%x, out_addr=%x"%(bufsize, outsize, in_addr,out_addr))
    return buf_map, in_addr,out_addr

################################################################################
#class demo_Layer:
#    def __init__(self, network, idx):


# gen_demo_layer function judge pre-layer and next-layer and quant info, 
# select gen what layer class, and may generate muti-layers 
# return layer list, and min,max info
#def gen_demo_layer(network, idx, tl_type_list, meta_info):



################################################################################
class Quant_Layer:
    def __init__(self, network, idx, quant_func, dataset):
        logging.info("### init Quant_Layer")
        self.type       = EL_QUANTIZE
        self.typename   = "EL_QUANTIZE"
        layer           = network.all_layers[idx]
        shape           = layer._nodes[0].out_tensors[0].shape
        self.count      = get_tensor_size(shape)
        
        minv, maxv, _   = quant_func(network, layer, dataset)
        self.scale, self.bias = min_max_to_scale_bias(minv, maxv)
        self.memsize    = self.count*(4+1)
        self.outsize    = self.count
        logging.info("###quant layer: count=%d, sclale=%f, bias=%f"%(self.count,self.scale,self.bias))
    def to_kmodel(self, arg_oft, eight_bit_mode, buf_map):
        cparser = cstruct.cstruct()
        cparser.load(kmodel_def)
        layer_header = cparser.kpu_model_layer_header_t()
        layer_body = cparser.kpu_model_quantize_layer_argument_t()    
        # fill layer body
        layer_body.flags                = 0
        buf_map, layer_body.main_mem_in_address, layer_body.mem_out_address = \
            cal_in_out_addr(buf_map, self.outsize)
        layer_body.count                = self.count
        layer_body.scale                = self.scale
        layer_body.bias                 = self.bias
        # fill header
        layer_header.type               = EL_QUANTIZE
        layer_header.body_size          = len(layer_body)
        # header, bin, memsize, (s,b)
        return layer_header, layer_body.dumps(), buf_map, (self.scale, self.bias)

def gen_quant_layer(network, idx, tl_type_list, meta_info):  
    meta_info['is_inai'] = False   #not in ai ram now
    if meta_info['is_quant'] == False:
        raise RuntimeError("not support float gen_dequant_layer yet !")
    else:   #quant
        el_layer = DeQuant_Layer(network, idx, quant_func, dataset)
        return [el_layer], meta_info

################################################################################
class DeQuant_Layer:
    def __init__(self, network, idx, quant_func, dataset):
        logging.info("### init DeQuant_Layer")
        self.type       = EL_DEQUANTIZE
        self.typename   = "EL_DEQUANTIZE"
        layer           = network.all_layers[idx]
        shape           = layer._nodes[0].out_tensors[0].shape
        self.count      = get_tensor_size(shape)
        
        if (self.count > 256*1024):
            logging.warn("output>1MB data, we assume it is dbg usage, cut to 1MB")
            self.count = 256*1024
        
        minv, maxv, _   = quant_func(network, layer, dataset)
        self.scale, self.bias = min_max_to_scale_bias(minv, maxv)
        self.memsize    = self.count*(4+1)
        self.outsize    = self.count*4

        
        
        logging.info("###dequant layer: count=%d, sclale=%f, bias=%f"%(self.count,self.scale,self.bias))
    def to_kmodel(self, arg_oft, eight_bit_mode, buf_map):
        logging.info(buf_map)
        cparser = cstruct.cstruct()
        cparser.load(kmodel_def)
        layer_header = cparser.kpu_model_layer_header_t()
        layer_body = cparser.kpu_model_dequantize_layer_argument_t()    
        # fill layer body
        layer_body.flags                = 0
        buf_map, layer_body.main_mem_in_address, layer_body.main_mem_out_address = \
            cal_in_out_addr(buf_map, self.outsize)
        layer_body.count                = self.count
        layer_body.scale                = self.scale
        layer_body.bias                 = self.bias
        # fill header
        layer_header.type               = EL_DEQUANTIZE
        layer_header.body_size          = len(layer_body)
        # header, bin, memsize, (s,b)
        return layer_header, layer_body.dumps(), buf_map, (self.scale, self.bias)
    

def gen_dequant_layer(network, idx, tl_type_list, meta_info):  
    meta_info['is_inai'] = False   #not in ai ram now
    if meta_info['is_quant'] == False:
        raise RuntimeError("not support float gen_dequant_layer yet !")
    else:   #quant
        meta_info['is_quant'] = False
        el_layer = DeQuant_Layer(network, idx, meta_info['quant_func'], meta_info['dataset'])
        return [el_layer], meta_info

################################################################################
class GAP2D_Layer:
    def __init__(self, network, idx):
        logging.info("### init GAP2D_Layer")
        self.type       = EL_GLOBAL_AVERAGE_POOL2D
        self.typename   = "EL_GLOBAL_AVERAGE_POOL2D"
        layer           = network.all_layers[idx]
        shape           = layer._nodes[0].in_tensors[0].shape
        if len(shape)   != 4:
            raise RuntimeError("only support (xx ,xx, xx, xx) shape!")
        self.kernel_size= shape[1]*shape[2]
        self.channels   = shape[3]
        self.memsize    = self.kernel_size*self.channels*4+self.channels*4
        self.outsize    = self.channels*4
        logging.info("GAP2D_Layer: kernel_size %d, channels %d"%(
            self.kernel_size, self.channels))
    def to_kmodel(self, arg_oft, eight_bit_mode, buf_map):
        cparser = cstruct.cstruct()
        cparser.load(kmodel_def)
        layer_header = cparser.kpu_model_layer_header_t()
        layer_body = cparser.kpu_model_gap2d_layer_argument_t()    
        # fill layer body
        layer_body.flags                = 0
        buf_map, layer_body.main_mem_in_address, layer_body.main_mem_out_address = \
            cal_in_out_addr(buf_map, self.outsize)
        layer_body.kernel_size          = self.kernel_size
        layer_body.channels             = self.channels
        # fill header
        layer_header.type               = EL_GLOBAL_AVERAGE_POOL2D
        layer_header.body_size          = len(layer_body)
        # header, bin, memsize, (s,b)
        
        return layer_header, layer_body.dumps(), buf_map, (0, 0)
        
        
def gen_gavgpool2d_layer(network, idx, tl_type_list, meta_info):
    layer = network.all_layers[idx]
    meta_info['is_inai'] = False   #not in ai ram now
    if meta_info['is_quant']:    # we dequant it first, keep same with nncase behavior
        dequant_layer = DeQuant_Layer(network, idx-1, meta_info['quant_func'], meta_info['dataset'])
        gap2d_layer = GAP2D_Layer(network, idx)
        quant_layer = Quant_Layer(network, idx, meta_info['quant_func'], meta_info['dataset'])
        return [dequant_layer, gap2d_layer, quant_layer], meta_info
    else:
        gap2d_layer = GAP2D_Layer(network, idx)
        return [gap2d_layer], meta_info
        

################################################################################
class QuantReshape_Layer:
    def __init__(self, network, idx):
        logging.info("### init QuantReshape_Layer")
        self.type = EL_QUANTIZED_RESHAPE
        self.typename = "EL_QUANTIZED_RESHAPE"
        layer = network.all_layers[idx]
        shape = layer._nodes[0].out_tensors[0].shape
        self.shape = shape
        logging.info("QuantReshape_Layer: shape to {}".format(shape))
        
class Reshape_Layer:
    def __init__(self, network, idx):
        self.type = EL_RESHAPE
        self.typename = "EL_RESHAPE"
        layer = network.all_layers[idx]
        shape = layer._nodes[0].out_tensors[0].shape
        self.shape = shape
        logging.info("Reshape_Layer: shape to {}".format(shape))

        
        

def gen_reshape_layer(network, idx, tl_type_list, meta_info):
    layer = network.all_layers[idx]
    meta_info['is_inai'] = False   #not in ai ram now
    quant_func = meta_info['quant_func']
    if meta_info['is_quant'] == False:
        meta_info['minv'], meta_info['maxv'], _ = quant_func(network, layer, meta_info['dataset'])
        el_layer = Reshape_Layer(network, idx)
        return [el_layer], meta_info
    else:   #quant
        meta_info['minv'], meta_info['maxv'], _ = quant_func(network, layer, meta_info['dataset'])
        el_layer = QuantReshape_Layer(network, idx)
        return [el_layer], meta_info

################################################################################
class QuantFlatten_Layer:
    def __init__(self, network, idx):
        logging.info("### init QuantFlatten_Layer")
        self.type = EL_QUANTIZED_TENSORFLOW_FLATTEN
        self.typename = "EL_QUANTIZED_TENSORFLOW_FLATTEN"
        layer = network.all_layers[idx]
        shape = layer._nodes[0].in_tensors[0].shape
        if len(shape) != 4:
            raise RuntimeError("only support 4-D flatten now!")
        self.shape = shape.as_list()
        self.memsize = self.shape[1]*self.shape[2]*self.shape[3]
        self.outsize = self.shape[1]*self.shape[2]*self.shape[3]
        logging.info("Flatten_Layer: w %d, h %d, ch %d"%(
            self.shape[1], self.shape[2], self.shape[3]))
    def to_kmodel(self, arg_oft, eight_bit_mode, buf_map):
        cparser = cstruct.cstruct()
        cparser.load(kmodel_def)
        layer_header = cparser.kpu_model_layer_header_t()
        layer_body = cparser.kpu_model_tf_flatten_layer_argument_t()    
        # fill layer body
        layer_body.flags                = 0
        buf_map, layer_body.main_mem_in_address, layer_body.main_mem_out_address = \
            cal_in_out_addr(buf_map, self.outsize)
        layer_body.width                = self.shape[1]
        layer_body.height               = self.shape[2]
        layer_body.channels             = self.shape[3]
        # fill header
        layer_header.type               = self.type
        layer_header.body_size          = len(layer_body)
        # header, bin, memsize, (s,b)
        return layer_header, layer_body.dumps(), buf_map, (0, 0)
            
def gen_flatten_layer(network, idx, tl_type_list, meta_info):
    meta_info['is_inai'] = False   #not in ai ram now
    if meta_info['is_quant'] == False:
        raise RuntimeError("only support quant flatten!")
    else:   #quant
        el_layer = QuantFlatten_Layer(network, idx)
        return [el_layer], meta_info


################################################################################
class QuantMaxPool2D_Layer:
    def __init__(self, network, idx):
        logging.info("### init QuantMaxPool2D_Layer")
        self.type = EL_QUANTIZED_MAX_POOL2D
        self.typename = "EL_QUANTIZED_MAX_POOL2D"
        layer = network.all_layers[idx]
        in_shape = layer._nodes[0].in_tensors[0].shape
        if len(in_shape) != 4:
            raise RuntimeError("only support 4-D flatten now!")
        out_shape = layer._nodes[0].out_tensors[0].shape
        if len(out_shape) != 4:
            raise RuntimeError("only support 4-D flatten now!")
        filter_size         = layer.layer_args['filter_size']
        strides             = layer.layer_args['strides']
        padding             = layer.layer_args['padding']
        self.in_shape       = in_shape.as_list()
        self.out_shape      = out_shape.as_list()
        self.kernel_width   = filter_size[0]
        self.kernel_height  = filter_size[1]
        self.stride_width   = strides[0]
        self.stride_height  = strides[1]
        if padding == 'SAME':
            self.padding_width  = 0
            self.padding_height = 0
        else:
            raise RuntimeError("not support valid padding for maxpool")
        
        self.memsize = self.in_shape[1]*self.in_shape[2]*self.in_shape[3]+\
            self.out_shape[1]*self.out_shape[2]*self.out_shape[3]
        self.outsize = self.out_shape[1]*self.out_shape[2]*self.out_shape[3]
        logging.info("QuantMaxPool2D_Layer: in {} , out {}, kernel {}, stride {}".format(
            self.in_shape, self.out_shape, filter_size, strides))
    def to_kmodel(self, arg_oft, eight_bit_mode, buf_map):
        cparser = cstruct.cstruct()
        cparser.load(kmodel_def)
        layer_header = cparser.kpu_model_layer_header_t()
        layer_body = cparser.kpu_model_quant_max_pool2d_layer_argument_t()    
        # fill layer body
        layer_body.flags                = 0
        buf_map, layer_body.main_mem_in_address, layer_body.main_mem_out_address = \
            cal_in_out_addr(buf_map, self.outsize)
        layer_body.in_shape.width       = self.in_shape[1]
        layer_body.in_shape.height      = self.in_shape[2]
        layer_body.in_shape.channels    = self.in_shape[3]
        layer_body.out_shape.width      = self.out_shape[1]
        layer_body.out_shape.height     = self.out_shape[2]
        layer_body.out_shape.channels   = self.out_shape[3]
        layer_body.kernel_width         = self.kernel_width
        layer_body.kernel_height        = self.kernel_height
        layer_body.stride_width         = self.stride_width
        layer_body.stride_height        = self.stride_height
        layer_body.padding_width        = self.padding_width
        layer_body.padding_height       = self.padding_height        
        # fill header
        layer_header.type               = self.type
        layer_header.body_size          = len(layer_body)
        # header, bin, memsize, (s,b)
        return layer_header, layer_body.dumps(), buf_map, (0, 0)

def gen_maxpool2d_layer(network, idx, tl_type_list, meta_info):
    meta_info['is_inai'] = False   #not in ai ram now
    if meta_info['is_quant'] == False:
        raise RuntimeError("only support quant flatten!")
    else:   #quant
        mp2d_layer = QuantMaxPool2D_Layer(network, idx)
        return [mp2d_layer], meta_info

################################################################################
class FullyConnected_Layer:
    def __init__(self, network, idx):
        logging.info("### init FullyConnected_Layer")
        #KLA_LINEAR = 0,KLA_RELU = 1,KLA_RELU6 = 2 
        self.type           = EL_FULLY_CONNECTED
        self.typename       = "EL_FULLY_CONNECTED"
        layer = network.all_layers[idx]
        in_shape = layer._nodes[0].in_tensors[0].shape.as_list()
        if len(in_shape)    != 2:
            raise RuntimeError("only support 2-D fc now!")
        out_shape = layer._nodes[0].out_tensors[0].shape.as_list()
        if len(out_shape)   != 2:
            raise RuntimeError("only support 2-D fc now!")
        act_table={'linear':0, 'relu':1, 'relu6':2}
        if 'act' not in layer.layer_args:
            act             = 'linear'
        else:
            act             = layer.layer_args['act']
        self.in_channels    = in_shape[1]
        self.out_channels   = out_shape[1]
        if act not in act_table:
            raise RuntimeError("FullyConnected_Layer not support %s now!"%act)
        logging.info("act=%s"%act)
        self.act            = act_table[act]
        self.W              = layer.W.numpy()
        self.b              = layer.b.numpy()
        
        self.memsize = self.in_channels*4 + self.out_channels*4
        self.outsize = self.out_channels*4
        logging.info("FullyConnected_Layer: in %d , out %d"%(self.in_channels, self.out_channels))
    def to_kmodel(self, arg_oft, eight_bit_mode, buf_map):
        cparser = cstruct.cstruct()
        cparser.load(kmodel_def)
        layer_header = cparser.kpu_model_layer_header_t()
        layer_body = cparser.kpu_model_fully_connected_layer_argument_t()    
        # fill layer body
        layer_body.flags                = 0
        buf_map, layer_body.main_mem_in_address, layer_body.main_mem_out_address = \
            cal_in_out_addr(buf_map, self.outsize)
        layer_body.in_channels          = self.in_channels
        layer_body.out_channels         = self.out_channels
        layer_body.act                  = self.act    
        # weights
        layer_weights   = bytearray()
        layer_bias      = bytearray()
        weights         = self.W
        bias            = self.b
        for i in range(self.out_channels):
            for j in range(self.in_channels):
                layer_weights += struct.pack("<f",weights[j][i])
            layer_bias += struct.pack("<f",bias[i])
        layer_bin = layer_body.dumps()+layer_weights+layer_bias
        
        # fill header
        layer_header.type               = self.type
        layer_header.body_size          = len(layer_bin)
        # header, bin, memsize, (s,b)
        #logging.info(layer_body)
        #logging.info(layer_body.dumps())
        return layer_header, layer_bin, buf_map, (0, 0)


def gen_fc_layer(network, idx, tl_type_list, meta_info):
    meta_info['is_inai'] = False   #not in ai ram now
    if meta_info['is_quant']:
        dequant_layer = DeQuant_Layer(network, idx-1, meta_info['quant_func'], meta_info['dataset'])
        fc_layer = FullyConnected_Layer(network, idx)
        quant_layer = Quant_Layer(network, idx, meta_info['quant_func'], meta_info['dataset'])
        return [dequant_layer, fc_layer, quant_layer], meta_info
    else:   #quant
        el_layer = FullyConnected_Layer(network, idx)
        return [el_layer], meta_info

################################################################################
class Upload_Layer:
    def __init__(self, network, idx):
        logging.info("### init Upload_Layer")
        #KLA_LINEAR = 0,KLA_RELU = 1,KLA_RELU6 = 2 
        self.type           = EL_K210_UPLOAD
        self.typename       = "EL_K210_UPLOAD"
        layer = network.all_layers[idx]
        shape = layer._nodes[0].out_tensors[0].shape.as_list()
        if len(shape)    != 4:
            raise RuntimeError("only support 4-D fc now!")
        self.width          = shape[1]
        self.height         = shape[2]
        self.channels       = shape[3]
        
        self.memsize = self.width*self.height*self.channels
        self.outsize = 0
        logging.info("Upload_Layer: WxHxC=%dx%dx%d"%(self.width, self.height, self.channels))
    def to_kmodel(self, arg_oft, eight_bit_mode, buf_map):
        cparser = cstruct.cstruct()
        cparser.load(kmodel_def)
        layer_header = cparser.kpu_model_layer_header_t()
        layer_body = cparser.kpu_model_upload_layer_argument_t()    
        # fill layer body
        layer_body.flags                = 0
        buf_map, layer_body.main_mem_in_address, _ = \
            cal_in_out_addr(buf_map, self.outsize)
        layer_body.kpu_mem_out_address  = 0 #upload, we just choose 0
        layer_body.width                =self.width        
        layer_body.height               =self.height       
        layer_body.channels             =self.channels     
        
        # fill header
        layer_header.type               = self.type
        layer_header.body_size          = len(layer_body)

        return layer_header, layer_body.dumps(), buf_map, (0, 0)

def gen_upload_layer(network, idx, tl_type_list, meta_info):
    meta_info['is_inai'] = True   #not in ai ram now
    if meta_info['is_quant']:
        upload_layer = Upload_Layer(network, idx)
        return [dequant_layer], meta_info
    else:   #quant
        raise RuntimeError("only support upload quant data to ai ram!")


################################################################################
class SoftMax_Layer:
    def __init__(self, network, idx):
        logging.info("### init Upload_Layer")
        #KLA_LINEAR = 0,KLA_RELU = 1,KLA_RELU6 = 2 
        self.type           = EL_SOFTMAX
        self.typename       = "EL_SOFTMAX"
        layer = network.all_layers[idx]
        shape = layer._nodes[0].out_tensors[0].shape.as_list()
        if len(shape)      != 2:
            raise RuntimeError("only support 1-D softmax now!")
        self.channels       = shape[1]
        self.memsize        = self.channels*4*2
        self.outsize        = self.channels*4
        logging.info("SoftMax: channels=%d"%(self.channels))
    def to_kmodel(self, arg_oft, eight_bit_mode, buf_map):
        cparser = cstruct.cstruct()
        cparser.load(kmodel_def)
        layer_header = cparser.kpu_model_layer_header_t()
        layer_body = cparser.kpu_model_softmax_layer_argument_t()    
        # fill layer body
        layer_body.flags                = 0
        buf_map, layer_body.main_mem_in_address, layer_body.main_mem_out_address = \
                                        cal_in_out_addr(buf_map, self.outsize)     
        layer_body.channels             =self.channels     
        # fill header
        layer_header.type               = self.type
        layer_header.body_size          = len(layer_body)

        return layer_header, layer_body.dumps(), buf_map, (0, 0)

def gen_softmax_layer(network, idx, tl_type_list, meta_info):
    meta_info['is_inai'] = True   #not in ai ram now
    if meta_info['is_quant']:
        raise RuntimeError("only support float data softmax now!")
    else:   #quant
        softmax_layer = SoftMax_Layer(network, idx)
        return [softmax_layer], meta_info


################################################################################
class Quant_Conv_Layer:
    def __init__(self, network, idx, tl_type_list, meta_info):
        logging.info("### init Quant_Conv_Layer")
        self.type = EL_CONV
        self.typename = "EL_CONV"
        layer = network.all_layers[idx]
        if meta_info['quant_bit'] != 8:
            raise RuntimeError("only support 8bit quant!")
        if conv_layer.layer_args['dilation_rate'] != (1,1):
            raise ValueError('only support (1,1) dilation_rate!')
        if conv_layer.layer_args['data_format'] != 'channels_last':
            raise ValueError('only support channels_last data_format!')  
        
        self.in_shape = layer._nodes[0].in_tensors[0].shape
        self.out_shape = layer._nodes[0].out_tensors[0].shape
        self.memsize = round_size_8(get_tensor_size(self.in_shape) + \
            get_tensor_size(self.out_shape))
        self.kernel = layer.config['args']['filter_size']
        self.stride = layer.config['args']['strides']
        self.filter_mult = layer.config['args']['n_filter']
        self.padding_type = layer.config['args']['padding']
        self.bias, self.weights, self.output_shift, self.bias_shift = [0,0,0,0] #TODO
        if self.padding_type == 'SAME':
            self.padw = (self.kernel[0] - 1) // 2
            self.padh = (self.kernel[1] - 1) // 2
        return 
        
def gen_conv_layer(network, idx, tl_type_list, meta_info):  
    if len(tl_type_list) != 1:
        raise ValueError('normal conv not include BN!')
    layer = network.all_layers[idx]
    if meta_info['is_quant'] == False:
        raise RuntimeError("not support float gen_conv_layer yet !")
    else:   #quant
        meta_info['minv'], meta_info['maxv'], _ = quant_func(network, layer, meta_info['dataset'])
        el_layer = Quant_Conv_Layer(network, idx, tl_type_list, meta_info['quant_info'])
        return [el_layer], meta_info

################################################################################
class Quant_DWConv_Layer:
    def __init__(self, network, idx, tl_type_list, meta_info):
        logging.info("### init Quant_DWConv_Layer")
        self.type = EL_DWCONV
        self.typename = "EL_DWCONV"
        layer = network.all_layers[idx]
        if meta_info['quant_bit'] != 8:
            raise RuntimeError("only support 8bit quant!")
        if layer.layer_args['dilation_rate'] != (1,1):
            raise ValueError('only support (1,1) dilation_rate!')
        if layer.layer_args['data_format'] != 'channels_last':
            raise ValueError('only support channels_last data_format!')  
        
        self.in_shape = layer._nodes[0].in_tensors[0].shape
        self.out_shape = layer._nodes[0].out_tensors[0].shape
        self.memsize = round_size_8(get_tensor_size(self.in_shape) + \
            get_tensor_size(self.out_shape))
        self.kernel = layer.config['args']['filter_size']
        self.stride = layer.config['args']['strides']
        self.filter_mult = layer.config['args']['n_filter']
        self.padding_type = layer.config['args']['padding']
        self.bias, self.weights, self.output_shift, self.bias_shift = [0,0,0,0] #TODO
        if self.padding_type == 'SAME':
            self.padw = (self.kernel[0] - 1) // 2
            self.padh = (self.kernel[1] - 1) // 2
        return 
        
def gen_dwconv_layer(network, idx, tl_type_list, meta_info):  
    if len(tl_type_list) != 1:
        raise ValueError('normal conv not include BN!')
    layer = network.all_layers[idx]
    if meta_info['is_quant'] == False:
        raise RuntimeError("not support float gen_dwconv_layer yet !")
    else:   #quant
        meta_info['minv'], meta_info['maxv'], _ = quant_func(network, layer, meta_info['dataset'])
        el_layer = Quant_DWConv_Layer(network, idx, tl_type_list, meta_info['quant_info'])
        return [el_layer], meta_info

################################################################################
def gen_add_layer(network, idx, tl_type_list, meta_info):
    raise RuntimeError("not support gen_add_layer yet !")
def gen_quant_add_layer(network, idx, tl_type_list, meta_info):
    raise RuntimeError("not support gen_quant_add_layer yet !")
def gen_gmaxpool2d_layer(network, idx, tl_type_list, meta_info):
    raise RuntimeError("not support gen_gmaxpool2d_layer yet !")
def gen_quant_gmaxpool2d_layer(network, idx, tl_type_list, meta_info):
    raise RuntimeError("not support gen_quant_gmaxpool2d_layer yet !")
def gen_avgpool2d_layer(network, idx, tl_type_list, meta_info):
    raise RuntimeError("not support gen_avgpool2d_layer yet !")
def gen_quant_avgpool2d_layer(network, idx, tl_type_list, meta_info):
    raise RuntimeError("not support gen_quant_avgpool2d_layer yet !")
def gen_requant_layer(network, idx, tl_type_list, meta_info):
    raise RuntimeError("not support gen_requant_layer yet !")
def gen_l2norm_layer(network, idx, tl_type_list, meta_info):
    raise RuntimeError("not support gen_l2norm_layer yet !")
def gen_concat_layer(network, idx, tl_type_list, meta_info):
    raise RuntimeError("not support gen_concat_layer yet !")
def gen_quant_concat_layer(network, idx, tl_type_list, meta_info):
    raise RuntimeError("not support gen_quant_concat_layer yet !")
def gen_quant_fc_layer(network, idx, tl_type_list, meta_info):
    raise RuntimeError("not support gen_quant_fc_layer yet !")
def gen_bn_layer(network, idx, tl_type_list, meta_info):  
    raise RuntimeError("not support gen_bn_layer yet !")
    
    

    

        
    

    
    