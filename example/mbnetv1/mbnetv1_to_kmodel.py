import tensorlayer as tl
import tensorflow as tf
from tensorlayer.models import Model
import maixemc as emc

from mobilenetv1 import MobileNetV1

tl.logging.set_verbosity(tl.logging.DEBUG)
mobilenetv1 = MobileNetV1(pretrained=True)
emc.save_kmodel(mobilenetv1, './mbnetv1.kmodel', './mbnetv1_dataset', dataset_func='img_0_1', quant_func='minmax', quant_bit=8, version=3)
