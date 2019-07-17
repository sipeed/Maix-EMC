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


import os
import numpy as np
from PIL import Image
import random
import tensorlayer as tl
from tensorlayer import logging



def box_image(im_path, new_w, new_h, new_ch):
    im = tl.vis.read_image(im_path)
    im = tl.prepro.imresize(im, (new_w, new_h)) / 255
    #im = im.astype(np.float32)[np.newaxis, ...]
    return im

def load_img_0_1(dataset_dir, w, h, ch):
    if os.path.isdir(dataset_dir):
        all_files = os.listdir(dataset_dir)
        if len(all_files) > 128:
            logging.info("[*] too many data in dataset, we random choose 128 datas {}")
            all_files = random.sample(all_files, 128)  # set maxmum dataset size

        dataset_file_list = [
            os.path.join(dataset_dir, f)
            for f in all_files
            if os.path.isfile(os.path.join(dataset_dir, f))
        ]
    else:
        dataset_file_list = (dataset_dir,)
        
    dataset_val = np.array([box_image(path, w, h, ch) for path in dataset_file_list], dtype=np.float32)
    return dataset_val

def load_img_0_255(dataset_dir, w, h, ch):
    dataset = load_img_0_1(dataset_dir, w, h, ch)
    return dataset * 255
    
def load_img_neg1_1(dataset_dir, w, h, ch):
    dataset = load_img_0_1(dataset_dir, w, h, ch)
    return dataset * 2 - 1   

loader_func_name_dict={
    'img_0_1':load_img_0_1,
    'img_0_255':load_img_0_255,
    'img_neg1_1':load_img_neg1_1}
    
def loader_func_valid(name):
    if name in loader_func_name_dict:
        return True
    else:
        return False
        
def loader_func_byname(name):
    return loader_func_name_dict[name]

def available_loader():
	for item in loader_func_name_dict:
		logging.info(item)
    
    
    

