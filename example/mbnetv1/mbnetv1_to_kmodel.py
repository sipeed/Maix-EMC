import tensorlayer as tl
import tensorflow as tf
from tensorlayer.models import Model
import maixemc as emc
import os

from mobilenetv1 import MobileNetV1

tl.logging.set_verbosity(tl.logging.DEBUG)
alpha=0.75
kmodel_name="./mbnetv1.kmodel"
kfpkg_name="./mbnetv1.kfpkg"
mobilenetv1 = MobileNetV1(pretrained=True, alpha=0.75)
emc.save_kmodel(mobilenetv1, kmodel_name, './mbnetv1_dataset', dataset_func='img_0_1', quant_func='minmax', quant_bit=8, version=3, sm_flag=True)
os.system('zip '+kfpkg_name+' '+kmodel_name+' flash-list.json')
