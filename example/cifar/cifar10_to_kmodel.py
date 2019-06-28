import tensorlayer as tl
from tensorlayer.models import Model
import maixemc as emc
import tensorlayer as tl

tl.logging.set_verbosity(tl.logging.DEBUG)
net = Model.load('./cifar10.h5', load_weights=True)
emc.save_kmodel(net, './cifar10.kmodel', './cifar10_dataset', dataset_func='img_0_1', quant_func='minmax', quant_bit=8, version=3)
