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


# Quant functions for Maix-EMC
# You can add your quant function in quant_func_name_dict
# 
# Quant function format:
# Input:   
#     Network:    TensorLayer Model 
#     Layer:      layer object
#     dataset:    dataset for Quant
#     is_weights: is quant for weight?
# Output:
#     minv:       minimum value in network's this layer's output
#     maxv:       maximum value in network's this layer's output
#     val:        batch vals


import numpy as np
from tensorlayer import logging

def QuantMinMax(network, layer, dataset, is_weights=False):
    if is_weights:
        weights = layer.all_weights[0].numpy()
        minv = weights.min()
        maxv = weights.max()
        return minv, maxv, weights
    else:
        network.eval()
        network(dataset)
        out = layer._nodes[0].out_tensors[0]
        val = out.numpy().flatten()
        minv = val.min()
        maxv = val.max()
        return minv, maxv, val
 
def QuantMinMax98(network, layer, dataset, is_weights=False):
    if is_weights:
        weights = layer.all_weights[0].numpy()
        minv = weights.min()
        maxv = weights.max()
        return minv, maxv, weights
    else:
        network.eval()
        network(dataset)
        out = layer._nodes[0].out_tensors[0]
        val = out.numpy().flatten()
        val_s = sorted(val)
        assert (len(val_s) >= 50)
        minv = val_s[round(len(val_s) * 0.01)]
        maxv = val_s[round(len(val_s) * 0.99)]
        return minv, maxv, val 


def QuantMeanMinMax(network, layer, dataset, is_weights=False):
    if is_weights:
        return QuantMinMax(network, layer, dataset, is_weights=True)
    else:
        network.eval()
        network(dataset)
        out = layer._nodes[0].out_tensors[0]
        val = out.numpy()
        val = np.reshape(val, [val.shape[0], np.prod(val.shape[1:])])
        minv = val.min(axis=1).mean()
        maxv = val.max(axis=1).mean()
        return minv, maxv, val
 
 
 
def QuantKLD(network, layer, dataset, is_weights=False):
    BINS_NUMBER = 8192
    QUANTIZE_SIZE = 256
    def chunks( l, n):
        """Yield successive n-sized chunks from l."""
        for i in range(0, len(l), n):
            yield l[i:i + n]

    def smooth( y, box_pts):
        box = np.ones(box_pts) / box_pts
        y_smooth = np.convolve(y, box, mode='same')
        return y_smooth

    def quantize_x( origin, x):
        chunked_data = list(chunks(origin, len(origin) // x))

        foo = [sum(i) for i in chunked_data]
        final_array = []

        for m, piece in enumerate(chunked_data):

            weight = foo[m]
            if weight == 0:
                final_array += [0] * len(piece)
                continue

            binary_piece = np.array(piece > 0)
            replace_val = foo[m] / sum(binary_piece)
            final_array += list(replace_val * binary_piece)

        return final_array

    def calc_kld( P, start_bin_max, end_bin_max, start_bin_min, end_bin_min, delta, max_val, min_val):
        import scipy.stats
        from copy import deepcopy
        klds = {}
        for i in range(start_bin_max, end_bin_max + 1, QUANTIZE_SIZE):
            for j in range(start_bin_min, end_bin_min + 1, QUANTIZE_SIZE):
                reference_distribution_P = deepcopy(P[j:i])
                left_outliers_count = np.sum(P[0:j])
                right_outliers_count = np.sum(P[i:BINS_NUMBER])

                reference_distribution_P[0] += left_outliers_count
                reference_distribution_P[-1] += right_outliers_count

                candidate_distribution_Q = quantize_x(reference_distribution_P, QUANTIZE_SIZE)
                left_outliers_P = deepcopy(P[:j + (i - j) // QUANTIZE_SIZE])
                right_outliers_P = deepcopy(P[i - (i - j) // QUANTIZE_SIZE:])
                left_replace_val = 0
                if sum(left_outliers_P > 0) > 0:
                    left_replace_val = sum(left_outliers_P) / sum(left_outliers_P > 0)
                right_replace_val = 0
                if sum(right_outliers_P > 0) > 0:
                    right_replace_val = sum(right_outliers_P) / sum(right_outliers_P > 0)
                candidate_distribution_Q = list(left_replace_val * (left_outliers_P > 0)) + candidate_distribution_Q[(
                                                                                                                                 i - j) // QUANTIZE_SIZE:i - j - (
                            i - j) // QUANTIZE_SIZE] + list(right_replace_val * (right_outliers_P > 0))

                Q = np.array(candidate_distribution_Q)

                kld = scipy.stats.entropy(P, Q)

                # logging.info((j,i), kld, (j + 0.5) * delta + (min_val - delta), (i + 0.5) * delta + (min_val - delta))
                klds[(j, i)] = kld

        return klds

    def convert_layer_output( data):
        image_num = data.shape[0]

        max_all = np.max(data)
        min_all = np.min(data)
        delta = (max_all - min_all) / (BINS_NUMBER + 1)
        bins_all = np.arange(min_all, max_all, delta)  # fixed bin size

        P = np.zeros(BINS_NUMBER)
        for image_idx in range(image_num):
            data_curr_image = np.ndarray.flatten(data[image_idx])

            n, bins = np.histogram(data, bins=bins_all)
            P = P + n

        return (P, min_all, max_all, delta)

    def find_min_max_kld( data):
        (P, min_data, max_data, delta) = convert_layer_output(data)
        P = smooth(P, 512)
        # find max first
        klds_max = calc_kld(P, QUANTIZE_SIZE, BINS_NUMBER, 0, 0, delta, max_data, min_data)
        (tmp, max_bin) = min(zip(klds_max.values(), klds_max.keys()))[1]
        klds_min = calc_kld(P, max_bin, max_bin, 0, max_bin - 1, delta, max_data, min_data)
        (min_bin, tmp) = min(zip(klds_min.values(), klds_min.keys()))[1]

        threshold_min = (min_bin) * delta + (min_data)
        threshold_max = (max_bin) * delta + (min_data)
        logging.info('Min data(threshold_min): %f'%threshold_min)
        logging.info('Max data(threshold_max): %f'%threshold_max)

        return (threshold_min, threshold_max)

    if is_weights:
        return QuantMinMax(network, layer, dataset, is_weights=True)
    else:
        network.eval()
        network(dataset)
        out = layer._nodes[0].out_tensors[0]
        val = out.numpy()
        minv, maxv = find_min_max_kld(val)
        return minv, maxv, val
 
      
quant_func_name_dict={
    'minmax':QuantMinMax,
    'minmax98':QuantMinMax98,
    'meanminmax':QuantMeanMinMax,
    'kld':QuantKLD,
    }
    
def quant_func_valid(name):
    if name in quant_func_name_dict:
        return True
    else:
        return False
        
def quant_func_byname(name):
    return quant_func_name_dict[name]

def available_quant():
	for item in quant_func_name_dict:
		logging.info(item)
