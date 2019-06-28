# Maix-Model-Convertor
Maix-EMC: Embedded/Edge Model Convertor, convert NN model for embedded systems


Maix-EMC act as **TensorLayer**（[https://github.com/tensorlayer/tensorlayer](https://github.com/tensorlayer/tensorlayer)）plugin, convert TensorLayer Model to kmodel.


**Kmodel** is an flatten embedded Model format, currently used on Maix K210 boards, but it is also possible to run on normal MCU by using corresponding driver.


We design Maix-EMC as a easy **extendable** frame, you can easily add new layer type or new MCU or new model convertor for it, detail refer to following sections or this post: https://bbs.sipeed.com/t/topic/916


![](https://git.kancloud.cn/repos/zepan/note/raw/f60065399babe7b3ce6fde5491b0467bc77a12ac/images/screenshot_1561694286461.png?access-token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJleHAiOjE1NjE3NDc3MzAsImlhdCI6MTU2MTcwNDUzMCwicmVwb3NpdG9yeSI6InplcGFuXC9ub3RlIiwidXNlciI6eyJ1c2VybmFtZSI6InplcGFuIiwibmFtZSI6InplcGFuIiwiZW1haWwiOiJjYWVzYXJAamljbS5jbiIsInRva2VuIjoiMzRjNDZkODFiNzllMTI2YTgwZTIzNzJlNDc2ZjVhNDAiLCJhdXRob3JpemUiOnsicHVsbCI6dHJ1ZSwicHVzaCI6dHJ1ZSwiYWRtaW4iOnRydWV9fX0.AetIzXNoxv3rYkZPTqu1GkJAIN4ZMxRkdY7WW2QjkTQ)

## How to use
### install packages
~~~
pip install dissect.cstruct 
~~~
install TensorLayer, version above 2.0.1: https://github.com/tensorlayer/tensorlayer

### usage
Refer to example/cifar

prepare some pics for quant, put into cifar_dataset floder

prepare the model you need convert to kmodel: cifar10.h5

Do like cifar10_to_kmodel.py:

~~~
emc.save_kmodel(net, './cifar10.kmodel', './cifar10_dataset', dataset_func='img_0_1', quant_func='minmax', quant_bit=8, version=3)
~~~

defination of save_kmodel:
~~~
save_kmodel(network, filepath, dataset_dir, dataset_func='img_0_1', quant_func='minmax', quant_bit=8, version=3):
~~~

and it will generate cifar10.kmodel
~~~
I0628 11:22:10.987743 139933152151360 tl_logging.py:95] generate layer 8: EL_FULLY_CONNECTED
I0628 11:22:10.987811 139933152151360 tl_logging.py:95] buf_map=[40960, 1, 0]
I0628 11:22:10.995115 139933152151360 tl_logging.py:95] buf_size=a000, outsize=28, in_addr=0, out_addr=9fd8
I0628 11:22:10.995439 139933152151360 tl_logging.py:95] max_mem_usage=40960 Byte, 40 KB
I0628 11:22:10.995719 139933152151360 tl_logging.py:99] ===============================end gen_kmodel===================================
I0628 11:22:10.995798 139933152151360 tl_logging.py:99] kmodel size = 283924 Byte, 277 KB
I0628 11:22:10.996294 139933152151360 tl_logging.py:99] [*] Saved TL model into ./cifar10.kmodel, dataset path=./cifar10_dataset, quant func is minmax, quant bit is 8, kmodel version is 3
~~~
Now, we use 1.bmp in dataset to compare PC calculation and K210 calculate:

(you can use MaixPy to execute cifar10.kmodel, or use the example project in the example/cifar10)
~~~
PC:
 1.0859082, -5.415024 ,  1.554703 ,  0.12335867,  0.46049562,
-0.6250827, -3.6137981,  0.5201204,  -0.6762826, -3.7476015 
K210:
 1.181304,  -5.381746,   1.580029,   0.045343,    0.511492, 
 -0.656629, -3.597826,   0.561016,   -0.733270,  -3.771462
~~~
Their result is close, but still need optimize quant.

## How to add new layer support
add your Layer class in edge_layer.py

## How to add new platform support
make new floder like 'stm32', and write stm32_constant.py,  stm32_layer.py(optimize  ops for stm32 layers), stm32_xxxmodel_bin.py.

## Files Description
### edge_model.py
It generate internal network layer list for model bin generator, layers in list have enough information to build anyother model,

and to_xxxmodel method will return bin of this layer, you can implement your self model method by adding to_xxxmodel method.  
~~~
def gen_edge_model(network, platform, version, dataset_dir, dataset_func, quant_func, quant_bit)
    generate edge model file from network, you can add platform in platform_table
    
def gen_edge_layers_from_network(network, platform, dataset, quant_func, quant_bit=8)
    generate layer list from network, corresponding platform layer map is in tl_to_xxx_table
    this table map   TL layer class to EMC layer generator, and contain TL layer combine information.

def gen_edge_layer_from_network(network, platform, meta_info, idx)
    generate single layer from TL layer
    all layer type in edge_constant
    all layer have init, to_xxxmodel method
~~~
### edge_constants.py
all EMC layer types, convert TL layer class to EMC layer types.


### edge_dataloader.py
load dataset from dir, generate ndarray for quant and infer usage.

you can add load method in loader_func_name_dict

### edge_quant.py
return min,max for quant reference.

you can add your minmax functions in quant_func_name_dict.

now support minmax and kld method.

### edge_layer.py
All common used layer (no kpu/npu/simd/dsp accelerate ops) is implemnet here.

every TL layer correspond to one or more EMC layer, gen_xxx_layer will do this map, it return EMC layer list of current TL layer.

EMC layer type defined in edge_constants.py, and implement in edge_layer.py and xxx_layer.py.

class xxx_Layer have at least two method:
~~~
def __init__(self, network, idx):
def to_kmodel(self, arg_oft, eight_bit_mode, buf_map):    #generate kmodel bin and other information
~~~

### k210_constant.py
all structure in k210's kpu.c

we use dissect.cstruct  parse c struct to python objects.

### k210_layer.py
contains kpu accelerate ops.

we transport orignal k210 kpu driver, so this file contains some warpper.

in normal mcu, you can redefine thoss layer to use ARM NN libs.

### k210_kmodel_bin.py
package every layer's bin to entirety kmodel.

 
