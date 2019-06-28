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


EL_DUMMY                            = -1
EL_INVALID 							= 0
EL_ADD								= 1
EL_QUANTIZED_ADD					= 2
EL_GLOBAL_MAX_POOL2D				= 3
EL_QUANTIZED_GLOBAL_MAX_POOL2D		= 4
EL_GLOBAL_AVERAGE_POOL2D			= 5
EL_QUANTIZED_GLOBAL_AVERAGE_POOL2D	= 6
EL_MAX_POOL2D						= 7
EL_QUANTIZED_MAX_POOL2D				= 8
EL_AVERAGE_POOL2D					= 9
EL_QUANTIZED_AVERAGE_POOL2D			= 10
EL_QUANTIZE							= 11
EL_DEQUANTIZE						= 12
EL_REQUANTIZE						= 13
EL_L2_NORMALIZATION					= 14
EL_SOFTMAX							= 15
EL_CONCAT							= 16
EL_QUANTIZED_CONCAT					= 17
EL_FULLY_CONNECTED					= 18
EL_QUANTIZED_FULLY_CONNECTED		= 19
EL_TENSORFLOW_FLATTEN				= 20
EL_QUANTIZED_TENSORFLOW_FLATTEN		= 21

EL_CONV                             =1000
EL_DWCONV                           =1001
EL_QUANTIZED_RESHAPE				=1002
EL_RESHAPE      	    			=1003

EL_K210_CONV 						= 10240
EL_K210_ADD_PADDING			        = 10241
EL_K210_REMOVE_PADDING			    = 10242
EL_K210_UPLOAD					    = 10243
