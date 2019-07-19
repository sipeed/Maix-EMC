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
 
kpu_layer_config_reg_offset = {
    'interrupt_enabe': 0,
    'image_addr': 1,
    'image_channel_num': 2,
    'image_size': 3,
    'kernel_pool_type_cfg': 4,
    'kernel_load_cfg': 5,
    'kernel_offset': 6,
    'kernel_calc_type_cfg': 7,
    'write_back_cfg': 8,
    'conv_value': 9,
    'conv_value2': 10,
    'dma_parameter': 11
}

kpu_layer_config_field_offset = {
    'interrupt_enabe': {
        'int_en': 0,
        'ram_flag': 1,
        'full_add': 2,
        'depth_wise_layer': 3
    },
    'image_addr': {
        'image_src_addr': 0,
        'image_dst_addr': 32
    },
    'image_channel_num': {
        'i_ch_num': 0,
        'o_ch_num': 32,
        'o_ch_num_coef': 48
    },
    'image_size': {
        'i_row_wid': 0,
        'i_col_high': 10,
        'o_row_wid': 32,
        'o_col_high': 42
    },
    'kernel_pool_type_cfg': {
        'kernel_type': 0,
        'pad_type': 3,
        'pool_type': 4,
        'first_stride': 8,
        'bypass_conv': 9,
        'load_para': 10,
        'dma_burst_size': 16,
        'pad_value': 24,
        'bwsx_base_addr': 32
    },
    'kernel_load_cfg': {
        'load_coor': 0,
        'load_time': 1,
        'para_size': 15,
        'para_start_addr': 32
    },
    'kernel_offset': {
        'coef_column_offset': 0,
        'coef_row_offset': 4
    },
    'kernel_calc_type_cfg': {
        'channel_switch_addr': 0,
        'row_switch_addr': 16,
        'coef_size': 20,
        'coef_group': 28,
        'load_act': 31,
        'active_addr': 32
    },
    'write_back_cfg': {
        'wb_channel_switch_addr': 0,
        'wb_row_switch_addr': 16,
        'wb_group': 20
    },
    'conv_value': {
        'shr_w': 0,
        'shr_x': 4,
        'arg_w': 8,
        'arg_x': 32
    },
    'conv_value2': {
        'arg_add': 0
    },
    'dma_parameter': {
        'send_data_out': 0,
        'channel_byte_num': 16,
        'dma_total_byte': 32
    }
}

kmodel_def ="""
typedef struct
{
    uint32 version;			
    uint32 flags;
    uint32 arch;				
    uint32 layers_length;		
    uint32 max_start_address;
    uint32 main_mem_usage;	
    uint32 output_count;		
} kpu_model_header_t;
typedef struct
{
    uint32 address;
    uint32 size;
} kpu_model_output_t;
typedef struct
{
    uint32 type;
    uint32 body_size;
} kpu_model_layer_header_t;
typedef struct
{
    uint32 width;
    uint32 height;
    uint32 channels;
} kpu_model_shape_t;
typedef struct
{
    uint32 flags;
    uint32 main_mem_out_address;
    uint32 layer_offset;
    uint32 weights_offset;
    uint32 bn_offset;
    uint32 act_offset;
} kpu_model_conv_layer_argument_t;
typedef struct
{
    uint32 flags;
    uint32 main_mem_in_address;
    uint32 main_mem_out_address;
    uint32 kernel_size;
    uint32 channels;
} kpu_model_gap2d_layer_argument_t;
typedef struct
{
    uint32 flags;
    uint32 main_mem_in_address;
    uint32 kpu_mem_out_address;
    uint32 width;
    uint32 height;
    uint32 channels;
} kpu_model_upload_layer_argument_t;
typedef struct
{
    uint32 flags;
    uint32 main_mem_in_address;
    uint32 mem_out_address;
    uint32 count;
    float scale;
	float bias;
} kpu_model_quantize_layer_argument_t;
typedef struct
{
    uint32 flags;
    uint32 main_mem_in_address;
    uint32 main_mem_out_address;
    uint32 count;
    float scale;
	float bias;
} kpu_model_dequantize_layer_argument_t;
typedef struct
{
    uint32 flags;
    uint32 main_mem_in_address;
    uint32 kpu_mem_out_address;
    uint32 channels;
} kpu_model_add_padding_layer_argument_t;
typedef struct
{
    uint32 flags;
    uint32 main_mem_in_address;
    uint32 main_mem_out_address;
    uint32 channels;
} kpu_model_remove_padding_layer_argument_t;
typedef struct
{
    uint32 flags;
    uint32 main_mem_in_address;
    uint32 main_mem_out_address;
    uint32 width;
    uint32 height;
    uint32 channels;
} kpu_model_tf_flatten_layer_argument_t;
typedef struct
{
    uint32 flags;
    uint32 main_mem_in_address;
    uint32 main_mem_out_address;
    kpu_model_shape_t in_shape;
    kpu_model_shape_t out_shape;
    uint32 kernel_width;
    uint32 kernel_height;
    uint32 stride_width;
    uint32 stride_height;
    uint32 padding_width;
    uint32 padding_height;
} kpu_model_quant_max_pool2d_layer_argument_t;
typedef struct
{
    uint32 flags;
    uint32 main_mem_in_address;
    uint32 main_mem_out_address;
    uint32 in_channels;
    uint32 out_channels;
    uint32 act;
} kpu_model_fully_connected_layer_argument_t;
typedef struct
{
    uint32 flags;
    uint32 main_mem_in_address;
    uint32 main_mem_out_address;
    uint32 channels;
} kpu_model_softmax_layer_argument_t;
"""