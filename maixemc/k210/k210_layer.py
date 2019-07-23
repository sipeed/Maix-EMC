# coding=utf-8
'''
 * Copyright 2019 Sipeed Inc.
 * Copyright 2018 Canaan Inc.
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
import numpy as np
from tensorlayer import logging

from .k210_constants import *
from dissect import cstruct 

import sys
sys.path.append("..")
from ..edge_constants import *
from ..edge_layer import *

default_conv_arg = None
default_act_arg = None
default_bn_arg = {
    'load_para': 0,
    'bwsx_base_addr': 0
}
default_pool_arg = {
    'pool_type': 0,  # bypass
}

###############################################################################
def signed_to_hex(value, width):
    return hex(int(round((1 << width) + value)) % (1 << width))


def min_max_to_scale_bias(minv, maxv):
    scale = (maxv - minv) / 255
    bias = minv
    return scale, bias

#def signed_to_hex(value, width):
#    if isinstance(value, np.ndarray):
#        value = value.tolist()
#    return hex(int(round((1 << width) + value)) % (1 << width))

def pow_next_log_of_2_no_round(value, bound_shift, shift_max_shift=4):
    mul, shift = np.frexp(np.abs(value))    # value = mul(0~1)*(1 << shift)
    ret = bound_shift - 1 - shift           # shift to full bound_shift
    mul = np.sign(value) * mul * np.power(2, bound_shift - 1)   # scale mul
    # value = mul>>ret
    return ret, mul

def pow_next_log_of_2(value, bound_shift, shift_max_shift=4):
    ret = 0     #limit shr < shift_max_shift
    shift_max = 1 << shift_max_shift
    while value >= -(1 << (bound_shift - 2)) and value < (1 << (bound_shift - 2)) \
            and value != 0 and ret < (shift_max - 1):
        value = value * 2   
        ret = ret + 1
    return ret, value
    
def pow_next_log_of_2_with_maxshift(value, bound_shift, shift_max):
    ret = 0     #limit shr < shift_max_shift
    while value >= -(1 << (bound_shift - 2)) and value < (1 << (bound_shift - 2)) \
            and value != 0 and ret < (shift_max - 1):
        value = value * 2   
        ret = ret + 1
    return ret, value
    
def align_4(oft):
    oft_align = ((((oft-1)>>2)+1)<<2)
    pad_len = oft_align - oft
    return [oft_align, pad_len]

def align_8(oft):
    oft_align = ((((oft-1)>>3)+1)<<3)
    pad_len = oft_align - oft
    return [oft_align, pad_len]

def align_128(oft):
    oft_align = ((((oft-1)>>7)+1)<<7)
    pad_len = oft_align - oft
    return [oft_align, pad_len]
###############################################################################
def gen_layer_struct(elayer, idx):
    reserved = 0
    set_to_zero = 0
    img_ram_size = 2 * 1024 * 1024

    # we do following ops in order, every ops give scale to next step
    # conv ops scale weight and x, so give swsx to next step
    conv_arg = elayer.conv and elayer.conv.to_kmodel_param() or default_conv_arg
    # bn ops scale b, so give swsxsb to next step
    bn_arg = elayer.bn and elayer.bn.to_kmodel_param(conv_arg['swsx'], conv_arg['scale_w_fix'], ) or default_bn_arg
    act_arg = elayer.act and elayer.act.to_kmodel_param(bn_arg['post_scale']) or default_act_arg
    pool_arg = elayer.pool and elayer.pool.to_kmodel_param() or default_pool_arg
    io_arg = elayer.to_kmodel_io_param()

    mino, maxo = elayer.act.min_y, elayer.act.max_y
    output_scale, output_bias = min_max_to_scale_bias(mino, maxo)

    img_input_size = int(math.ceil(io_arg['i_ch_num'] / conv_arg['coef_group']) * 64 * conv_arg['channel_switch_addr'])
    img_output_size = int(math.ceil(io_arg['o_ch_num'] / io_arg['wb_group']) * 64 * io_arg['wb_channel_switch_addr'])

    assert (img_input_size + img_output_size <= img_ram_size)
    
    logging.debug("----KPU Register Config Info----")
    interrupt_enabe = {
        'int_en': set_to_zero,
        'ram_flag': reserved,
        'full_add': set_to_zero,
        'depth_wise_layer': conv_arg['depth_wise_layer']
    }
    logging.debug("       {}".format(interrupt_enabe))
    image_addr = {
        'image_src_addr': hex(int((0 if not idx & 1 else (img_ram_size - img_input_size)) / 64)),
        'image_dst_addr': hex(int((0 if idx & 1 else (img_ram_size - img_output_size)) / 64))
    }
    logging.debug("       {}".format(image_addr))
    image_channel_num = {
        'i_ch_num': hex(io_arg['i_ch_num'] - 1),
        'o_ch_num': hex(io_arg['o_ch_num'] - 1),
        'o_ch_num_coef': hex(conv_arg['o_ch_num_coef'] - 1),
    }
    logging.debug("       {}".format(image_channel_num))
    image_size = {
        'i_row_wid': hex(conv_arg['i_row_wid'] - 1),
        'i_col_high': hex(conv_arg['i_col_high'] - 1),
        'o_row_wid': hex(io_arg['o_row_wid'] - 1),
        'o_col_high': hex(io_arg['o_col_high'] - 1),
    }
    logging.debug("       {}".format(image_size))
    kernel_pool_type_cfg = {
        'kernel_type': conv_arg['kernel_type'],
        'pad_type': conv_arg['pad_type'],
        'pool_type': pool_arg['pool_type'],
        'first_stride': conv_arg['first_stride'],
        'bypass_conv': 0 if elayer.conv else 1,
        'load_para': bn_arg['load_para'],
        'dma_burst_size': io_arg['dma_burst_size'],
        'pad_value': signed_to_hex(conv_arg['pad_value'], 8),
        'bwsx_base_addr': bn_arg['bwsx_base_addr'],
    }
    
    kernel_pool_type_cfg_print = {
        'kernel_type': conv_arg['kernel_type'],
        'pad_type': conv_arg['pad_type'],
        'pool_type': pool_arg['pool_type'],
        'first_stride': conv_arg['first_stride'],
        'bypass_conv': 0 if elayer.conv else 1,
        'load_para': bn_arg['load_para'],
        'dma_burst_size': io_arg['dma_burst_size'],
        'pad_value': signed_to_hex(conv_arg['pad_value'], 8),
        'bwsx_base_addr': 'too many content',
    }
    logging.debug("       {}".format(kernel_pool_type_cfg_print))
    kernel_load_cfg = {
        'load_coor': conv_arg['load_coor'],
        'load_time': conv_arg['load_time'] - 1,
        'para_size': conv_arg['para_size'],
        'para_start_addr': conv_arg['para_start_addr'],
    }
    
    kernel_load_cfg_print = {
        'load_coor': conv_arg['load_coor'],
        'load_time': conv_arg['load_time'] - 1,
        'para_size': conv_arg['para_size'],
        'para_start_addr': 'too many content',
    }
    logging.debug("       {}".format(kernel_load_cfg_print))
    kernel_offset = {
        'coef_column_offset': set_to_zero,
        'coef_row_offset': set_to_zero,
    }
    logging.debug("       {}".format(kernel_offset))
    kernel_calc_type_cfg = {
        'channel_switch_addr': hex(conv_arg['channel_switch_addr']),
        'row_switch_addr': hex(conv_arg['row_switch_addr']),
        'coef_size': reserved,
        'coef_group': conv_arg['coef_group'],
        'load_act': 1 if elayer.act else 0,
        'active_addr': act_arg['active_addr']
    }
    
    kernel_calc_type_cfg_print = {
        'channel_switch_addr': hex(conv_arg['channel_switch_addr']),
        'row_switch_addr': hex(conv_arg['row_switch_addr']),
        'coef_size': reserved,
        'coef_group': conv_arg['coef_group'],
        'load_act': 1 if elayer.act else 0,
        'active_addr': 'too many content'
    }
    logging.debug("       {}".format(kernel_calc_type_cfg_print))
    write_back_cfg = {
        'wb_channel_switch_addr': hex(io_arg['wb_channel_switch_addr']),
        'wb_row_switch_addr': hex(io_arg['wb_row_switch_addr']),
        'wb_group': io_arg['wb_group']
    }
    logging.debug("       {}".format(write_back_cfg))
    conv_value = {
        'shr_w': conv_arg['shr_w'],
        'shr_x': conv_arg['shr_x'],
        'arg_w': signed_to_hex(conv_arg['arg_w'], 24),
        'arg_x': signed_to_hex(conv_arg['arg_x'], 24),
    }
    logging.debug("       {}".format(conv_value))
    conv_value2 = {
        'arg_add': int(round(conv_arg['arg_add'])),
    }
    logging.debug("       {}".format(conv_value2))
    dma_parameter = {
        'send_data_out': io_arg['send_data_out'],
        'channel_byte_num': io_arg['channel_byte_num'] - 1,
        'dma_total_byte': io_arg['dma_total_byte'] - 1,
    }
    logging.debug("       {}".format(dma_parameter))

    return {
        'interrupt_enabe': interrupt_enabe,
        'image_addr': image_addr,
        'image_channel_num': image_channel_num,
        'image_size': image_size,
        'kernel_pool_type_cfg': kernel_pool_type_cfg,
        'kernel_load_cfg': kernel_load_cfg,
        'kernel_offset': kernel_offset,
        'kernel_calc_type_cfg': kernel_calc_type_cfg,
        'write_back_cfg': write_back_cfg,
        'conv_value': conv_value,
        'conv_value2': conv_value2,
        'dma_parameter': dma_parameter
    }, (output_scale, output_bias)

def gen_layer_code(elayer_struct, layer_cfg):
    layer_cfg.reg_arg = bytearray(len(elayer_struct[0].items()) * 8)
    for reg_name, data in elayer_struct[0].items():
        value = 0
        for filed_name, filed_value in data.items():
            #logging.info("  %s: %s"%(filed_name, filed_value))
            if isinstance(filed_value, int):
                pass
            elif isinstance(filed_value, str):
                filed_value = int(filed_value, 16) if '0x' in filed_value else int(filed_value)
            else:
                filed_value = 0
            value |= (filed_value << kpu_layer_config_field_offset[reg_name][filed_name])
        value&=0xffffffffffffffff
        layer_cfg.reg_arg[kpu_layer_config_reg_offset[reg_name]*8:kpu_layer_config_reg_offset[reg_name]*8+8] = value.to_bytes(8, 'little')


def gen_bn_code(elayer_struct, layer_cfg):
    bn_list = elayer_struct[0]['kernel_pool_type_cfg']['bwsx_base_addr']
    layer_cfg.bn_len = len(bn_list) * 8
    layer_cfg.bn_arg = bytearray(layer_cfg.bn_len)
    i = 0
    for bn in bn_list:
            layer_cfg.bn_arg[i:i+8] = (int(bn['norm_mul'], 16) + (int(bn['norm_add'], 16) << 24) + (int(bn['norm_shift']) << 56)).to_bytes(8, 'little')
            i += 8

def gen_act_code(elayer_struct, layer_cfg):
    act_list = elayer_struct[0]['kernel_calc_type_cfg']['active_addr']
    for item in act_list:
        layer_cfg.act_arg += (int(item['dxs']) + (int(item['dy']) << 8) + (int(signed_to_hex(item['x'], 36), 16) << 24)).to_bytes(8, 'little')
    bias_list = [int(item['y']) for item in act_list]
    value1, value2 = 0, 0
    for index in range(8):
        value1 += (bias_list[index] << (8 * index))
        value2 += (bias_list[index + 8] << (8 * index))
    layer_cfg.act_arg += value1.to_bytes(8, 'little')
    layer_cfg.act_arg += value2.to_bytes(8, 'little')

def gen_weights_code(elayer_struct, layer_cfg, eight_bit_mode):
    weights = elayer_struct[0]['kernel_load_cfg']['para_start_addr']
    if eight_bit_mode:
        layer_cfg.weights_len = len(weights)
        layer_cfg.weights_arg = bytearray(layer_cfg.weights_len)
        i = 0
        for item in weights:
            layer_cfg.weights_arg[i:i+1] = int(signed_to_hex(item, 8), 16).to_bytes(1, 'little')
            i += 1
    else:
        layer_cfg.weights_len = len(weights) * 2
        layer_cfg.weights_arg = bytearray(layer_cfg.weights_len)
        i = 0
        for item in weights:
            layer_cfg.weights_arg[i:i+2] = int(signed_to_hex(item, 16), 16).to_bytes(2, 'little')
            i += 2


class layer_config_struct():
    def __init__(self):
        self.reg_addr_offset = 0
        self.reg_arg = b''
        self.act_addr_offset = 0
        self.act_arg = b''
        self.bn_addr_offset = 0
        self.bn_len = 0
        self.bn_arg = b''
        self.weights_addr_offset = 0
        self.weights_len = 0
        self.weights_arg = b''    
    
    
###############################################################################
class K210Conv:
    def __init__(self, weights, depth_wise_layer, eight_bit_mode, xy_shape, xw_minmax, quant_func):
        self.weights = weights
        self.weights_shape = self.weights.shape
        self.input_shape, self.output_shape = xy_shape
        xmin, xmax, wmin, wmax = xw_minmax
        self.stride = 1
        self.depth_wise_layer = depth_wise_layer
        self.eight_bit_mode = eight_bit_mode
        self.quant_func = quant_func

        self.wmax = wmax
        self.wmin = wmin
        self.x_range = xmax - xmin
        self.x_bias = xmin
        if self.x_range == 0:
            self.x_range = 0.00001

        if len(wmin.shape) == 0:
            self.is_chwise = False
            self.w_range = wmax - wmin
            self.w_bias = wmin
            if self.w_range == 0:
                self.w_range = 0.00001
            self.w_range_all = wmax-wmin
            self.w_bias_all = wmin
        else:
            self.is_chwise = True
            self.w_range_all = max(wmax)-min(wmin)
            self.w_bias_all = min(wmin)
            self.w_range = wmax - wmin
            self.w_bias = wmin
            if self.w_range_all == 0:
                self.w_range_all = 0.00001
            for _range in self.w_range:
                if _range == 0:
                    _range = 0.00001

        if self.input_shape[1] < 4:
            tensor_height = self.input_shape[1]
            logging.info('[error] feature map required height>4 which this layer height is {}' \
                  .format(tensor_height))
            self.input_shape = list(self.input_shape)
            self.output_shape = list(self.output_shape)
            old_input_wh = self.input_shape[1:3]
            old_output_wh = self.output_shape[1:3]
            self.input_shape[1:3] = [4, 4]
            self.output_shape[1:3] = [4, 4]
            notice = 'this layer heigh-width MUST padding from {}x{}=>{}x{} to 4x4=>4x4 in CPU before continue.' \
                .format(*old_input_wh, *old_output_wh)
            logging.info('[notice] ' + ('=' * 71))
            logging.info('[notice] ' + notice)
            logging.info('[notice] ' + ('=' * 71))
            raise ValueError('conv height must > 4' )

    @staticmethod
    def q(value, scale, bias):
        return (value - bias) / scale

    def para_mult_loads(self, weights_shape, output_shape, kernel_size):
        weight_buffer_size = 2 * 9 * 4096
        weights_ich = int(weights_shape[2])
        weights_och = int(weights_shape[3])
        weight_data_size = 1 if self.eight_bit_mode else 2

        if self.depth_wise_layer:
            o_ch_weights_size = int(weights_shape[0]) * int(weights_shape[1]) * weight_data_size
        else:
            o_ch_weights_size = int(weights_shape[0]) * int(weights_shape[1]) * int(weights_shape[2]) * weight_data_size

        if int(weights_shape[0]) == 1:
            o_ch_weights_size_pad = math.ceil(o_ch_weights_size / 8) * 9
        else:
            o_ch_weights_size_pad = o_ch_weights_size
            assert (int(weights_shape[0]) == 3)

        if kernel_size == 3:
            load_time = math.ceil(weights_och / math.floor(4096 * 2 / weight_data_size / weights_ich))
        elif kernel_size == 1:
            load_time = math.ceil(weights_och / math.floor(4096 * 8 * 2 / weight_data_size / weights_ich))
        else:
            load_time = None
            assert (None)

        o_ch_num = int(output_shape[3])
        o_ch_num_coef = math.floor(weight_buffer_size / o_ch_weights_size_pad)

        if self.eight_bit_mode:
            half_weight_buffer_size = weight_buffer_size / 2
            while True:
                last_ch_idx = (o_ch_num - 1) % o_ch_num_coef
                last_addr_end = (last_ch_idx + 1) * o_ch_weights_size_pad
                if last_addr_end < half_weight_buffer_size:
                    break

                o_ch_num_coef = o_ch_num_coef - 1
                load_time = math.ceil(o_ch_num / o_ch_num_coef)
                if o_ch_num_coef <= 0:
                    assert ('cannot fix last_addr_end to first half part')

        assert (load_time <= 64)

        o_ch_num_coef = min(o_ch_num_coef, o_ch_num)
        para_size = o_ch_num_coef * o_ch_weights_size
        return load_time, para_size, o_ch_num_coef

    def to_kmodel_param(self):
        input_shape = self.input_shape
        output_shape = self.output_shape
        weights_shape = self.weights_shape
        weights = self.weights.transpose([3, 2, 0, 1])
        stride = self.stride

        weight_data_size = 1 if self.eight_bit_mode else 2
        kernel_size = int(weights_shape[0])

        # img i
        i_row_wid = int(input_shape[2])
        i_col_high = int(input_shape[1])
        coef_group = 1 if i_row_wid > 32 else (2 if i_row_wid > 16 else 4)
        row_switch_addr = math.ceil(i_row_wid / 64)
        channel_switch_addr = i_col_high * row_switch_addr
        # conv
        depth_wise_layer = 1 if self.depth_wise_layer else 0
        kernel_type = {1: 0, 3: 1}[kernel_size]
        pad_type = 0
        load_coor = 1

        first_stride = 0 if stride == 1 else 1
        assert (256 >= (i_col_high if first_stride == 0 else i_col_high / 2))

        load_time, para_size, o_ch_num_coef = self.para_mult_loads(weights_shape, output_shape, kernel_size)

        x_qmax = 255
        w_qmax = (1 << (8 * weight_data_size)) - 1
        
        # scale channel weight to full range first
        if self.is_chwise:
            logging.debug("---- channel wise scale ----")
            wmax_all = max(self.wmax)
            wmin_all = min(self.wmin)
            scale_w  = np.zeros(weights.shape[0])
            for i in range(weights.shape[0]):
                s1 = wmax_all / self.wmax[i]
                s2 = wmin_all / self.wmin[i]
                s = max(s1, s2) if (s1 < 0 or s2 < 0) else min(s1, s2);
                if s <= 0:
                    raise ValueError("channel wise scale error!")
                weights[i] *= s
                scale_w[i] = s
                #logging.debug("ch %d: max: %.3f,%.3f; min: %.3f,%.3f; s1=%.3f,s2=%.3f; s=%.3f"%(i,wmax_all,self.wmax[i], wmin_all,self.wmin[i],s1,s2,s))
            # TODO: use quant_func
            wmin_all = weights.min()
            wmax_all = weights.max()
            self.w_range_all = wmax_all - wmin_all
            self.w_bias_all = wmin_all
            scale_w_fix = 1/scale_w
        else:
            scale_w_fix = 1
            
        
        bias_x, scale_x = self.x_bias, self.x_range / x_qmax
        bias_w_all, scale_w_all = self.w_bias_all, self.w_range_all / w_qmax

        bx_div_sx = bias_x / scale_x     
        bw_div_sw_all = bias_w_all / scale_w_all

        shr_x, arg_x = pow_next_log_of_2(bw_div_sw_all, 24) #bw_div_sw = arg_x >> shr_x
        shr_w, arg_w = pow_next_log_of_2(bx_div_sx, 24) #bx_div_sx = arg_w >> shr_w
        arg_add = kernel_size * kernel_size * bw_div_sw_all * bx_div_sx
        pad_value = 0 -bx_div_sx
        swsx = scale_w_all * scale_x

        logging.debug("---- Doing conv quant  x = xq*scale_x + bias_x ----")
        logging.debug("quant X: bias %f, range %f ---> bias %f, scale %f"% \
            (self.x_bias, self.x_range, bias_x, scale_x))
        logging.debug("quant W: bias %f, range %f ---> bias %f, scale %f"% \
            (self.w_bias_all, self.w_range_all, bias_w_all, scale_w_all))

        weight_q = ((weights - bias_w_all) / scale_w_all)
        para_start_addr = [int(round(item)) for item in np.reshape(weight_q, (np.product(weight_q.shape),))]
        #print(para_start_addr)

        return {
            'swsx': swsx,
            'coef_group': coef_group,
            'channel_switch_addr': channel_switch_addr,
            'depth_wise_layer': depth_wise_layer,
            'o_ch_num_coef': o_ch_num_coef,
            'i_row_wid': i_row_wid,
            'i_col_high': i_col_high,
            'kernel_type': kernel_type,
            'pad_type': pad_type,
            'first_stride': first_stride,
            'pad_value': pad_value,
            'load_coor': load_coor,
            'load_time': load_time,
            'para_size': para_size,
            'para_start_addr': para_start_addr,
            'row_switch_addr': row_switch_addr,
            'shr_w': shr_w,
            'shr_x': shr_x,
            'arg_w': arg_w,
            'arg_x': arg_x,
            'arg_add': arg_add, 
            'scale_w_fix': scale_w_fix
        }


class K210BN:
    def __init__(self, mean, var, gamma, beta, epsilon, eight_bit_mode):
        self.mean = mean
        self.var = var
        self.gamma = gamma
        self.beta = beta
        self.epsilon = epsilon
        self.eight_bit_mode = eight_bit_mode

    @staticmethod
    def get_bn(scale, shift, bias):
        norm_shift, norm_mul = shift, scale
        return {
            'norm_mul': signed_to_hex(norm_mul, 24),
            'norm_add': signed_to_hex(bias, 32),
            'norm_shift': norm_shift
        }

    def to_kmodel_param(self, swsx=1, scale_w_fix=1):
        rsqrt_var = 1.0 / np.sqrt(self.var + self.epsilon)
        scale = self.gamma * rsqrt_var  #8.775
        bias = self.beta - self.gamma * self.mean * rsqrt_var
        
        # todo: rewrite this, make max_abs mul is +-(1<<N)
        # now we need convert bias from float to 36bit int
        bmax = max(abs(np.min(scale)), abs(np.max(scale)))
        brange = bmax
        sb = brange / 255
        if np.min(scale) == np.max(scale):
            sb = 1
        swsxsb = swsx * sb
        load_para = 1
        
        act_shift = 10
        post_scale = np.power(2, act_shift)
        if type(scale_w_fix) == int: #not channel wise
            swsxsb = swsxsb*scale_w_fix*post_scale
            out_shift, out_mul = pow_next_log_of_2_with_maxshift(swsxsb, 22, 15+1) # out_mul>>out_shift
            bn_shift = np.ones(len(bias)) *(out_shift)
            scale = (scale / sb * out_mul).round().astype('int32')
            bias = (bias * post_scale ).round().astype('int32')
        else:   #channel wise
            swsxsb = swsxsb * scale_w_fix * post_scale
            #logging.debug("{}, {}".format(swsxsb, scale_w_fix))
            bn_shift = np.ones(len(bias))
            for i in range(len(bias)):
                out_shift, out_mul = pow_next_log_of_2_with_maxshift(swsxsb[i], 22, 15+1)
                bn_shift[i] = out_shift
                scale[i] = (scale[i] / sb * out_mul).round().astype('int32')
                bias[i] = (bias[i] * post_scale ).round().astype('int32')
                #logging.debug("ch %d: swsxsb=%f, shift=%d, mul=%f, bn_shift=%d,scale(17bit)=0x%x, bias=0x%x"%(i,swsxsb[i],out_shift, out_mul, bn_shift[i] , int(scale[i]), int(bias[i])))

        bwsx_base_addr = [
            self.get_bn(s, shift, b)
            for s,shift,b in zip(scale, bn_shift, bias)
        ]

        return locals()


class K210Act:
    def __init__(self, min_y, max_y, ty, eight_bit_mode):
        if isinstance(ty, list) or isinstance(ty, tuple):
            self.ty = ty[0]
            self.leaky_mul = ty[1]
        else:
            self.ty = ty
        self.eight_bit_mode = eight_bit_mode
        self.min_y = min_y
        self.max_y = max_y

    @staticmethod
    def leaky_relu(x, v_mul):
        return x if x >= 0 else x * v_mul

    @staticmethod
    def leaky_relu_inverse(y, v_mul):
        return y if y >= 0 else y / v_mul

    @staticmethod
    def relu_inverse(y):
        return y

    @staticmethod
    def relu6_inverse(y):
        return y

    @staticmethod
    def leaky_table(min_y, max_y, v_mul):
        range_y = max_y - min_y
        y_table = [min_y + i * range_y / 15 for i in range(15)]
        y_table.append(max_y)
        if 0 not in y_table:
            y_table.append(0)
        y_table = sorted(y_table)
        x_table = [K210Act.leaky_relu_inverse(it, v_mul) for it in y_table]
        dydx = [(y_table[i + 1] - y_table[i]) / (x_table[i + 1] - x_table[i]) for i in range(len(y_table) - 1)]
        return zip(x_table, y_table, dydx)

    @staticmethod
    def relu_table(min_y, max_y):
        range_y = max_y - min_y
        y_table = [min_y + i * range_y / 15 for i in range(15)]
        y_table.append(max_y)
        if 0 not in y_table:
            y_table.append(0)
        y_table = sorted(y_table)
        x_table = [K210Act.relu_inverse(it) for it in y_table]
        dydx = [(y_table[i + 1] - y_table[i]) / (x_table[i + 1] - x_table[i]) for i in range(len(y_table) - 1)]
        return zip(x_table, y_table, dydx)

    @staticmethod
    def relu6_table(min_y, max_y):
        range_y = max_y - min_y
        y_table = [min_y + i * range_y / 15 for i in range(15)]
        y_table.append(max_y)
        if 0 not in y_table:
            y_table.append(0)
        y_table = sorted(y_table)
        x_table = [K210Act.relu6_inverse(it) for it in y_table]
        dydx = [(y_table[i + 1] - y_table[i]) / (x_table[i + 1] - x_table[i]) for i in range(len(y_table) - 1)]
        return zip(x_table, y_table, dydx)

    @staticmethod
    def linear_table(min_y, max_y):
        range_y = max_y - min_y
        y_table = [min_y + i * range_y / 15 for i in range(15)]
        if 0 not in y_table:
            y_table.append(0)
        y_table.append(max_y)
        y_table = sorted(y_table)
        return zip(y_table, y_table, [1] * (len(y_table) - 1))

    @staticmethod
    def find_shift(dydx):
        ret_shift = 0
        while abs(dydx) < (1 << 14) and dydx > 0:
            dydx = dydx * 2
            ret_shift = ret_shift + 1
        return ret_shift, dydx

    @staticmethod
    def table_to_act(act_table, min_y, max_y, eight_bit_mode, post_scale):
        def act_table_aux(x, y, dydx):
            y_scale = (max_y - min_y) / 255
            y_bias = min_y
            x_fix = x * post_scale                  # x scale 
            y_fix = (y - y_bias) / y_scale          # y scale to 0~255
            dydx_fix = dydx / y_scale / post_scale  # slope, y/

            yf_q = round(y_fix)                     # y bias, 0~255
            yf_err = y_fix - yf_q                   # 
            xfy = x_fix - yf_err / dydx_fix         # fix y to x 
            return xfy, yf_q, dydx_fix              # xstart, y bias, y_mul>>shift

        act_table = [(0x800000000, 0, 0)] + [act_table_aux(x, y, dydx) for x, y, dydx in act_table]
        
        #logging.info(act_table)
        #logging.info("miny=%f,maxy=%f"%(min_y,max_y))
        def ret_aux(x, y, dydx):
            dxss, dys = K210Act.find_shift(dydx)
            assert (dys >= 0)
            return {'x': int(round(x)), 'y': int(round(y)), 'dxs': dxss, 'dy': int(round(dys))}

        return [ret_aux(x, y, dydx) for x, y, dydx in act_table]

    def to_kmodel_param(self, post_scale):
#tl act dict
# _act_dict = {
    # "relu": tf.nn.relu,
    # "relu6": tf.nn.relu6,
    # "leaky_relu": tf.nn.leaky_relu,
    # "lrelu": tf.nn.leaky_relu,
    # "softplus": tf.nn.softplus,
    # "tanh": tf.nn.tanh,
    # "sigmoid": tf.nn.sigmoid,
# }
        act_tab = None
        if self.ty == 'leaky_relu' or self.ty == 'lrelu' :
            act_tab = list(K210Act.leaky_table(self.min_y, self.max_y, self.leaky_mul))
        elif self.ty == 'relu':
            act_tab = list(K210Act.relu_table(self.min_y, self.max_y))
        elif self.ty == 'relu6':
            act_tab = list(K210Act.relu6_table(self.min_y, self.max_y))
        elif self.ty == 'linear':
            act_tab = list(K210Act.linear_table(self.min_y, self.max_y))
        else:
            assert ValueError(self.ty, ' active is not supported.')

        active_tab = K210Act.table_to_act(list(act_tab), self.min_y, self.max_y, self.eight_bit_mode, post_scale)

        return {'active_addr': active_tab[:16]}


class K210Pool:
    def __init__(self, pool_type, size, stride):
        self.size = size
        self.stride = stride
        self.pool_type = pool_type

    def to_kmodel_param(self):
        if self.pool_type == 'MaxPool':
            return {'pool_type': {
                (2, 2): 1,
                (4, 4): 3,
                (2, 1): 9
            }[(self.size, self.stride)]}
        elif self.pool_type == 'AvgPool':
            return {'pool_type': {
                (2, 2): 2,
                (4, 4): 4,
                (2, 1): 8
            }[(self.size, self.stride)]}
        elif self.pool_type == 'leftPool':
            return {'pool_type': {
                (2, 2): 5,
                (4, 4): 7,
            }[(self.size, self.stride)]}
        elif self.pool_type == 'rightPool':
            return {'pool_type': 6}
        else:
            return None


class K210_Conv_Layer:
    def __init__(self, iwo_minmax, ico_shapes, conv_weights_isdw, bn_mean_var_gamma_beta_epsilon, act_type,
                 pool_type_size_stride, conv_idx, output_en, quant_func, eight_bit_mode=False):
        logging.info("### init K210_Conv_Layer")
        input_min, input_max, weights_min, weights_max, output_min, output_max = iwo_minmax
        input_shape, conv_shape, output_shape = ico_shapes
        conv_weights, conv_isdw = conv_weights_isdw
        self.conv_idx = conv_idx
        # KPU consist of conv, pool, bn, act
        # Conv
        self.type = EL_K210_CONV
        self.typename = "EL_K210_CONV"
        self.conv = K210Conv(
            conv_weights,
            conv_isdw,
            eight_bit_mode, [input_shape, conv_shape],
            [input_min, input_max, weights_min, weights_max],
            quant_func
        )

        bn_mean, bn_var, bn_gamma, bn_beta, bn_epsilon = bn_mean_var_gamma_beta_epsilon
        self.bn = K210BN(
            bn_mean,
            bn_var,
            bn_gamma,
            bn_beta,
            bn_epsilon,
            eight_bit_mode,
        )

        self.act = K210Act(output_min, output_max, act_type, eight_bit_mode=eight_bit_mode)

        if pool_type_size_stride is not None:
            pool_type, pool_size, pool_stride = pool_type_size_stride
            if pool_size == 2 and conv_shape[3] % 2 != 0:
                raise ValueError(
                    "this layer unsupport padding mode SAME of pooling"
                )

            #if conv_isdw and pool_size != 1:
            #    raise ValueError(
            #        'this layer not supported DepthwiseConv2d followed by pooling witch pool_size is not 1.'
            #    )

            self.pool = K210Pool(pool_type, pool_size, pool_stride)
        else:
            self.pool = None
        
        self.output_en = output_en
        if output_en:
            self.memsize = output_shape[1]*output_shape[2]*output_shape[3]
            self.outsize = self.memsize
        else:
            self.memsize = 0    #no need normal mem
            self.outsize = self.memsize

    @staticmethod
    def batch(iter, n=1):
        l = len(iter)
        for ndx in range(0, l, n):
            yield iter[ndx:min(ndx + n, l)]

    def to_kmodel_io_param(self):
        output_shape = self.conv.output_shape

        weights_shape = self.conv.weights_shape
        input_shape = self.conv.input_shape
        i_row_wid = int(input_shape[1])
        img_data_size = 1

        coef_group = 1 if i_row_wid > 32 else (2 if i_row_wid > 16 else 4)

        # io
        i_ch_num = int(weights_shape[2])
        o_ch_num = int(output_shape[3])
        # img o
        o_row_wid = int(output_shape[2])
        o_col_high = int(output_shape[1])
        wb_group = 1 if o_row_wid > 32 else (2 if o_row_wid > 16 else 4)
        wb_row_switch_addr = math.ceil(o_row_wid / 64)
        wb_channel_switch_addr = o_col_high * wb_row_switch_addr
        channel_byte_num = o_row_wid * o_col_high

        int_en = 1 if self.output_en else 0
        image_src_addr = None
        image_dst_addr = None
        dma_total_byte = o_row_wid * o_col_high * o_ch_num
        dma_burst_size = 0xf
        send_data_out = 1 if self.output_en else 0
        return locals()
        
    def to_kmodel(self, arg_oft, eight_bit_mode, buf_map):
        struct = gen_layer_struct(self, self.conv_idx)
        output_scale, output_bias = struct[1]
        cparser = cstruct.cstruct()
        cparser.load(kmodel_def)
        layer_header = cparser.kpu_model_layer_header_t()
        layer_config = layer_config_struct()    #bin
        # gen some bins
        gen_layer_code(struct, layer_config)    #kpu reg info
        gen_act_code(struct, layer_config)      #act table
        gen_bn_code(struct, layer_config)       #bn table
        gen_weights_code(struct, layer_config, eight_bit_mode)
                                                #conv weights
        layer_arg = cparser.kpu_model_conv_layer_argument_t()
        layer_arg_size = len(layer_arg.dumps())
        act_len = len(layer_config.act_arg)     
        #fill layer arg param
        layer = layer_arg
        layer.flags = 1 if self.output_en else 0
        buf_map, _, layer.main_mem_out_address = cal_in_out_addr(buf_map, self.outsize)
        [layer.layer_offset, pad_layer] = align_8(arg_oft + layer_arg_size)
        [layer.weights_offset, pad_weights] = align_128(layer.layer_offset + 12*8)   
        [layer.bn_offset, pad_bn] = align_128(layer.weights_offset + layer_config.weights_len)  
        [layer.act_offset, pad_act] = align_128(layer.bn_offset + layer_config.bn_len)
       
        layer_bin = layer.dumps() + \
            (b'\0'*pad_layer) + layer_config.reg_arg + \
            (b'\0'*pad_weights) + layer_config.weights_arg + \
            (b'\0'*pad_bn) + layer_config.bn_arg + \
            (b'\0'*pad_act) + layer_config.act_arg
        layer_header.type = EL_K210_CONV
        layer_header.body_size = len(layer_bin)
        
        logging.info("###K210 Conv Layer @0x%x"%arg_oft)
        if layer.flags :
            logging.info("output en, main_mem_out_address = 0x%x"%(layer.main_mem_out_address))
        logging.info("layer_offset=0x%x, weights_offset=0x%x, bn_offset=0x%x, act_offset=0x%x"%(layer.layer_offset, layer.weights_offset, layer.bn_offset, layer.act_offset))
        
        return layer_header, layer_bin, buf_map, (output_scale, output_bias)
        

class K210_Upload_Layer:
    def __init__(self, network, idx):
        self.type       = EL_K210_UPLOAD
        self.typename   = "EL_K210_UPLOAD"
        layer           = network.all_layers[idx-1]
        shape           = layer._nodes[0].out_tensors[0].shape
        self.width      = shape[1]
        self.height     = shape[2]
        self.channel    = shape[3]
    
    
################################################################################
# in kpu.c, it is not add zeros, just put data to kpu ram in right oft
class AddPadding_Layer:
    def __init__(self, network, idx, tl_type_list, meta_info):
        logging.info("### init AddPadding_Layer")
        self.type       = EL_K210_ADD_PADDING
        self.typename   = "EL_K210_ADD_PADDING"
        layer           = network.all_layers[idx]
        shape           = layer._nodes[0].in_tensors[0].shape
        if len(shape)   != 4:
            raise ValueError('K210 only support 4-d input tensor!')
        self.channels   = shape[3]
        self.memsize    = shape[1]*shape[2]*shape[3]
        self.outsize    = 0     #put to kpu
        logging.debug("AddPadding_Layer, channel=%d"%self.channels)
    def to_kmodel(self, arg_oft, eight_bit_mode, buf_map):
        cparser = cstruct.cstruct()
        cparser.load(kmodel_def)
        layer_header = cparser.kpu_model_layer_header_t()
        layer_body = cparser.kpu_model_add_padding_layer_argument_t()    
        # fill layer body
        layer_body.flags                = 0
        _, layer_body.main_mem_in_address, _ = \
            cal_in_out_addr(buf_map, self.outsize)
        layer_body.kpu_mem_out_address  = 0
        
        buf_map['pingpong']             = 0
        buf_map['last_addr']            = 0
        
        layer_body.channels             = self.channels
        # fill header
        layer_header.type               = EL_K210_ADD_PADDING
        layer_header.body_size          = len(layer_body)
        # header, bin, memsize, (s,b)
        return layer_header, layer_body.dumps(), buf_map, (0, 0)
        
class RemovePadding_Layer:
    def __init__(self, network, idx, tl_type_list, meta_info, output_shape):
        logging.info("### init RemovePadding_Layer")
        self.type       = EL_K210_REMOVE_PADDING
        self.typename   = "EL_K210_REMOVE_PADDING"
        layer           = network.all_layers[idx]
        shape           = layer._nodes[0].out_tensors[0].shape
        if len(shape)   != 4:
            raise ValueError('K210 only support 4-d input tensor!')
        self.channels   = shape[3]   
        self.memsize    = output_shape[1]*output_shape[2]*output_shape[3]
        self.outsize    = output_shape[1]*output_shape[2]*output_shape[3]
        logging.debug("RemovePadding_Layer, channel=%d"%self.channels)
    def to_kmodel(self, arg_oft, eight_bit_mode, buf_map):
        cparser = cstruct.cstruct()
        cparser.load(kmodel_def)
        layer_header = cparser.kpu_model_layer_header_t()
        layer_body = cparser.kpu_model_remove_padding_layer_argument_t()    
        # fill layer body
        layer_body.flags                = 0
        buf_map, layer_body.main_mem_in_address, layer_body.main_mem_out_address = \
            cal_in_out_addr(buf_map, self.outsize)
        layer_body.channels             = self.channels
        # fill header
        layer_header.type               = EL_K210_REMOVE_PADDING
        layer_header.body_size          = len(layer_body)
        # header, bin, memsize, (s,b)
        return layer_header, layer_body.dumps(), buf_map, (0, 0)
        
################################################################################   
#NCHW       NHWC
#KL_K210_CONV                         :   [['Conv2d'],
#                                        ['Conv2d', 'BatchNorm'],
#                                        ['DepthwiseConv2d'],
#                                        ['DepthwiseConv2d', 'BatchNorm']],
def gen_k210_conv_layer(network, idx, tl_type_list, meta_info):  
    def min_max_to_scale_bias(minv, maxv):
        scale = (maxv - minv) / 255
        bias = minv
        return scale, bias  
    layer_list = []
    layers = network.all_layers  
    
    logging.debug("gen k210 conv layer from tl_type_list: {}".format(tl_type_list))
    
    #check if the padding is right
    if (tl_type_list[0] == 'ZeroPad2d'):    
        zeropad_layer = layers[idx]
        if zeropad_layer.layer_args['padding'] != ((1, 1), (1, 1)):
            raise ValueError('K210 assume use ((1, 1), (1, 1)) zero padding!' )
        #idx += 1                        # skip the padding layer
        conv_layer = layers[idx+1]
        padding = conv_layer.layer_args['padding'] 
        strides = conv_layer.layer_args['strides'] 
        if padding != 'VALID':
            raise ValueError("K210 assume conv layer after zeropad use padding = 'VALID'" )
        if strides != (2, 2):
            raise ValueError("K210 assume conv layer after zeropad use strides = (2,2)" )
        
        if len(tl_type_list)>2 :
            bn_layer = layers[idx+2]
        else:
            bn_layer = None
            
        conv_isdw = (tl_type_list[1] == 'DepthwiseConv2d')  
    else:
        conv_layer = layers[idx]
        padding = conv_layer.layer_args['padding'] 
        strides = conv_layer.layer_args['strides'] 
        if strides != (1, 1):
            raise ValueError("K210 assume conv layer which stride > (1,1) use ZeroPad2d ahead " )
        
        if len(tl_type_list)>1 :
            bn_layer = layers[idx+1]
        else:
            bn_layer = None
            
        conv_isdw = (tl_type_list[0] == 'DepthwiseConv2d')  

    # valid parm check
    if conv_layer.layer_args['dilation_rate'] != (1,1):
        raise ValueError('only support (1,1) dilation_rate!')
    if conv_layer.layer_args['data_format'] != 'channels_last':
        raise ValueError('only support channels_last data_format!')  
    if  conv_layer.layer_args['filter_size'] != (1,1) and \
        conv_layer.layer_args['filter_size'] != (3,3) :
        raise ValueError('only support 1x1 or 3x3 filter_size!')   
    if conv_layer.layer_args['layer_type'] != 'normal':
        raise ValueError('only support normal layer_type!')  
        
    # Conv2d or DepthwiseConv2d Layer
    input_shape = conv_layer._nodes[0].in_tensors[0].shape
    conved_shape = conv_layer._nodes[0].out_tensors[0].shape
    output_shape = conved_shape
    input_shape = input_shape.as_list()
    conved_shape = conved_shape.as_list()
    output_shape = output_shape.as_list()

    if (tl_type_list[0] == 'ZeroPad2d'):    #strip 
        input_shape[1] -= 2
        input_shape[2] -= 2
    
    if len(input_shape) != 4:
        raise ValueError('K210 only support 4-d input tensor!')
    if input_shape[3] > 1024:
        raise ValueError('K210 only support max 1024 channel feature!')
    
    small_conv_flag = 0
    if input_shape[2] < 4 or input_shape[1] < 4:
        logging.info("too small conv, padding it first!")
        small_conv_flag = 1
        addpadding_layer = AddPadding_Layer(network, idx, tl_type_list, meta_info)
        if input_shape[1] < 4:
            input_shape[1] = 4
            conved_shape[1] = 4
            output_shape[1] = 4
        if input_shape[2] < 4:
            input_shape[2] = 4
            conved_shape[2] = 4
            output_shape[2] = 4
            
    conv_weights = conv_layer.all_weights[0].numpy()
    if hasattr(conv_layer, 'b'):
        conv_bias = conv_layer.b.numpy()
    else:
        conv_bias = 0
    #weights_min, weights_max, _ = meta_info['quant_func'](network, conv_layer, meta_info['dataset'], is_weights=True, is_chwise=False)
    weights_min, weights_max, _ = meta_info['quant_func'](network, conv_layer, meta_info['dataset'], is_weights=True, is_chwise=(bn_layer != None) and (not conv_isdw))
    
    
    # Pool in Conv Layer
    stride = conv_layer.layer_args['strides'] 
    if stride[0] != stride[1]:
        raise ValueError('only support square stride !')
    pool_size = stride[0]   #square size
    pool_stride = stride[0] #stride step
    if stride != (1,1):
        pool_type = 'leftPool'
        pool_type_size_stride = [pool_type, pool_size, pool_stride]
    else:
        pool_type_size_stride = None

    if pool_size>1 and input_shape[1] % pool_size != 0:
        if conv_layer.layer_args['padding'] == 'SAME':
            raise ValueError("at {} unsupport padding mode SAME of pooling with size > 1".format(conv_layer.layer_args['name']))

    # BN Layer
    if bn_layer != None:
        bn_mean_var_gamma_beta_epsilon = [
            bn_layer.moving_mean.numpy().flatten()-(conv_bias),
            bn_layer.moving_var.numpy().flatten(),
            bn_layer.gamma.numpy().flatten(),
            bn_layer.beta.numpy().flatten(),
            bn_layer.epsilon,
        ]
    else:
        bn_mean_var_gamma_beta_epsilon = [
            np.zeros([conved_shape[3]]), np.ones([conved_shape[3]]), np.ones([conved_shape[3]]), np.zeros([conved_shape[3]]), np.zeros([conved_shape[3]])
        ]
        

        
    # Act Layer
    if (bn_layer != None):
        if (bn_layer.layer_args['act'] != None):
            act_min_y, act_max_y, _ = meta_info['quant_func'](network, bn_layer, meta_info['dataset'])
            act_type = bn_layer.layer_args['act']
        else :
            act_min_y, act_max_y, _ = meta_info['quant_func'](network, bn_layer, meta_info['dataset'])
            act_type = 'linear'
    else:   #no act, use linear to bypass
        act_min_y, act_max_y, _ = meta_info['quant_func'](network, conv_layer, meta_info['dataset'])
        act_type = 'linear'
    eight_bit_mode = (meta_info['quant_bit'] == 8)
    
    # is it need output to normal memory?
    output_en = False
    # if next layer fork to another branch, we should output it to ram
    if idx+len(tl_type_list) >= len(layers):
        output_en = True
    else: 
        next_layer = layers[idx+len(tl_type_list)]
        next_layer_config = next_layer.config
        next_shape = next_layer._nodes[0].in_tensors[0].shape
        this_layer_node = layers[idx+len(tl_type_list)-1].config['args']['name'] +'_node_0'
        if (next_layer_config['prev_layer'][0] != this_layer_node):    #fork branch
            output_en = True
        # if next layer is non conv layer, output it to ram
        elif (next_layer_config['class'] != 'Conv2d') and (next_layer_config['class'] != 'DepthwiseConv2d'):
            output_en = True
        # if next layer is conv layer, but use cpu calculate
        elif next_shape[1] < 4 or next_shape[2] < 4:
            output_en = True
    
    if small_conv_flag == 1:    #TODO: need download to cpu ram first
        logging.info("samll conv, need padding, conv_idx reset to 0")
        meta_info['conv_idx'] = 0
    else:   #k210 
        if meta_info['is_inai'] == False:  #not in ai ram, we need upload it first
            logging.info("need upload, conv_idx reset to 0")
            meta_info['conv_idx'] = 0
    kl_args = {
        'iwo_minmax': [meta_info['last_min'], meta_info['last_max'], weights_min, weights_max, act_min_y, act_max_y],
        'ico_shapes': [input_shape, conved_shape, output_shape],
        'conv_weights_isdw':[conv_weights, conv_isdw],
        'bn_mean_var_gamma_beta_epsilon': bn_mean_var_gamma_beta_epsilon,
        'act_type': act_type,
        'pool_type_size_stride':pool_type_size_stride,
        'conv_idx':meta_info['conv_idx'],
        'output_en':output_en,
        'quant_func' : meta_info['quant_func'],
        'eight_bit_mode': eight_bit_mode,
    }
    # fix some critical condition
    # kl_args_fixed = k210_layer_post_fix(kl_args)
    
    # logging.info layer info
    output_min = act_min_y
    output_max = act_max_y
    layer_shape_trans = [
        int(input_shape[1]), int(input_shape[2]), int(input_shape[3]),
        int(output_shape[1]), int(output_shape[2]), int(output_shape[3])
    ]
    if bn_layer != None:
        output_name = bn_layer.layer_args['name']
    else:
        output_name = conv_layer.layer_args['name']
    logging.info("in min:%f, max:%f;  out min %f, max: %f"%(meta_info['last_min'], meta_info['last_max'], output_min, output_max))
    input_scale, input_bias = min_max_to_scale_bias(meta_info['last_min'], meta_info['last_max'])
    output_scale, output_bias = min_max_to_scale_bias(output_min, output_max)

    logging.info("**********gen_conv_layer")
    logging.info('     shape(HWC): {}x{}x{} ==> {}x{}x{}'.format(*layer_shape_trans))
    logging.info('     scale,bias: ({},{}) ==> ({},{})'.format(input_scale, input_bias, output_scale, output_bias))
    
    #convert to k210 layer
    kconv_layer = K210_Conv_Layer(**kl_args)

    if small_conv_flag == 1:    #TODO: need download to cpu ram first
        removepadding_layer = RemovePadding_Layer(network, idx, tl_type_list, meta_info, output_shape)
        layer_list.append(addpadding_layer)
        layer_list.append(kconv_layer)
        layer_list.append(removepadding_layer)
        meta_info['conv_idx']   = 0
        meta_info['is_inai']    = False
        meta_info['last_min']   = act_min_y
        meta_info['last_max']   = act_max_y
    else:   #k210 
        if meta_info['is_inai'] == False:  #not in ai ram, we need upload it first
            upload_layer = Upload_Layer(network, idx-1)
            layer_list.append(upload_layer)
            layer_list.append(kconv_layer)
            meta_info['conv_idx']   = 0
            meta_info['is_inai']    = True
            meta_info['last_min']   = act_min_y
            meta_info['last_max']   = act_max_y
        else:   #in ai ram already
            layer_list.append(kconv_layer)
            meta_info['conv_idx']   += 1
            meta_info['is_inai']    = True
            meta_info['last_min']   = act_min_y
            meta_info['last_max']   = act_max_y
            
    return layer_list, meta_info


def k210_layer_post_fix(el_list):
    def expand_wh(shape_):
        shape_1 = shape_[1] * 2
        shape_2 = shape_[2] * 2
        return [shape_[0], shape_1, shape_2, shape_[3]]
    def fix_dw_with_strde2(el_list):
        lack_of_left_pooling = False
        last_is_conv = True
        for index in range(len(el_list)):
            el = el_list[index]
            logging.debug("Layer %d:"%index)
            if el.type == EL_K210_CONV :
                conv_part = el.conv
                pool_part = el.pool
                input_shape = conv_part.input_shape
                output_shape = conv_part.output_shape
                conv_shape = output_shape
                conv_weights = conv_part.weights
                conv_isdw = conv_part.depth_wise_layer
                if pool_part is not None:
                    pool_type_size_stride = [pool_part.pool_type, pool_part.size, pool_part.stride]
                else:
                    pool_type_size_stride = None
                conv_kernel_size = int(conv_weights.shape[0])
                conv_stride = int((int(input_shape[2]) + 1) / int(conv_shape[2]))

                logging.debug("conv_stride=%d, conv_isdw=%d, conv_kernel_size=%d, pool==None:%d"%(conv_stride, conv_isdw, conv_kernel_size, (pool_type_size_stride is None)))

                if lack_of_left_pooling:
                    if last_is_conv == False:
                        raise ValueError('run fix_dw_with_strde2 failed, last not conv layer')
                    if not conv_isdw and conv_kernel_size == 1 and pool_type_size_stride is None:
                        # fix in current layer
                        input_shape = expand_wh(input_shape)
                        conv_shape = expand_wh(conv_shape)
                        lack_of_left_pooling = False
                        el.pool = K210Pool('leftPool', 2, 2)
                        #el.conv.output_shape = conv_shape
                        el.conv.input_shape = input_shape
                        logging.debug("Fixed: in normal 1x1 conv")
                    else:
                        if not (conv_kernel_size == 1 and pool_type_size_stride is None):
                            raise ValueError(
                                'run fix_dw_with_strde2 failed. ' +
                                'can not delay left_pooling over current layer, ' +
                                'current layer conv_kernel_size:{}, pool_type_size_stride:{}' \
                                .format(conv_kernel_size, pool_type_size_stride)
                            )

                        # delay fix in after layers
                        input_shape = expand_wh(input_shape)
                        conv_shape = expand_wh(conv_shape)
                        output_shape = expand_wh(output_shape)
                        el.conv.input_shape = input_shape
                        #el.conv.output_shape = conv_shape
                        logging.debug("Fixed: in next dw 1x1 conv")

                if conv_stride == 2:
                    if not conv_isdw:
                        logging.debug("we have done before")
                    else:
                        # dw layer needs to fix it later, it is chip bug/feature
                        lack_of_left_pooling = True
                        el.pool = None
                        conv_shape = expand_wh(conv_shape)
                        output_shape = expand_wh(output_shape)
                        el.conv.output_shape = conv_shape
                        logging.debug("dw conv stride fix in later layer")
                elif conv_stride != 1:
                    raise ValueError('unsupported stride!')
                else :
                    logging.debug("stride == 1, nothing to do")
            else:
                last_is_conv = False
                logging.debug("Not Conv Layer")

        if lack_of_left_pooling:
            raise ValueError('run fix_dw_with_strde2 failed. no more layers for fix.')
        return 
    
    logging.debug("----Start fix stride----")
    fix_dw_with_strde2(el_list)
    logging.debug("----End fix stride----")
    return 
