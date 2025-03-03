import sys, os
import cv2
import numpy as np
import torch
import mxnet as mx
from mxnet import ndarray as nd
import copy
import re


class Loader_HDA_Doppelganger:
    def __init__(self):
        pass


    def get_all_files_in_path(self, folder_path, file_extension=['.jpg','.jpeg','.png'], pattern=''):
        def natural_sort(l):
            convert = lambda text: int(text) if text.isdigit() else text.lower()
            alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
            return sorted(l, key=alphanum_key)

        file_list = []
        for root, _, files in os.walk(folder_path):
            for filename in files:
                path_file = os.path.join(root, filename)
                for ext in file_extension:
                    if pattern in path_file and path_file.lower().endswith(ext.lower()):
                        file_list.append(path_file)
                        print(f'Found files: {len(file_list)}', end='\r')
        print()
        # file_list.sort()
        file_list = natural_sort(file_list)
        return file_list


    def load_images_protocol(self, data_dir=''):
        files_paths = self.get_all_files_in_path(data_dir)

        protocol = int(len(files_paths)/2) * [None]
        for i in range(0, len(files_paths), 2):
            sample0, sample1 = files_paths[i], files_paths[i+1]
            sample0_gender = sample0.split('/')[-2]
            sample1_gender = sample1.split('/')[-2]
            pair_label = 0

            pair = {}
            pair['sample0'] = sample0
            pair['sample0_gender'] = sample0_gender
            
            pair['sample1'] = sample1
            pair['sample1_gender'] = sample1_gender
            
            pair['pair_label'] = pair_label

            protocol[int(i/2)] = pair

        return protocol


    def load_dataset(self, data_dir, image_size, replace_ext='.png'):
        pairs_orig = self.load_images_protocol(data_dir)
        pairs_update = pairs_orig

        data_list = []
        for flip in [0, 1]:
            data = torch.empty((len(pairs_update)*2, 3, image_size[0], image_size[1]))
            data_list.append(data)

        issame_list               = np.array([bool(pairs_update[i]['pair_label']) for i in range(len(pairs_update))])
        gender_list               = np.array([sorted((pairs_update[i]['sample0_gender'], pairs_update[i]['sample1_gender'])) for i in range(len(pairs_update))])
        samples_orig_paths_list   = np.array([sorted((pairs_orig[i]['sample0'], pairs_orig[i]['sample1'])) for i in range(len(pairs_orig))])
        samples_update_paths_list = np.array([sorted((pairs_update[i]['sample0'], pairs_update[i]['sample1'])) for i in range(len(pairs_update))])
        
        for idx in range(len(pairs_update) * 2):
            idx_pair = int(idx/2)
            if idx % 2 == 0:
                img_path = pairs_update[idx_pair]['sample0']
            else:
                img_path = pairs_update[idx_pair]['sample1']
            assert os.path.isfile(img_path), f"Error, file not found: '{img_path}'"
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = mx.nd.array(img)

            if img.shape[1] != image_size[0]:
                img = mx.image.resize_short(img, image_size[0])
            img = nd.transpose(img, axes=(2, 0, 1))
            for flip in [0, 1]:
                if flip == 1:
                    img = mx.ndarray.flip(data=img, axis=2)
                data_list[flip][idx][:] = torch.from_numpy(img.asnumpy())
            if idx % 100 == 0:
                print(f"loading pairs {idx}/{len(pairs_update)*2}", end='\r')
        print('\n', data_list[0].shape)
        # return data_list, issame_list, races_list, subj_list, samples_orig_paths_list, samples_update_paths_list
        return data_list, issame_list, gender_list, samples_orig_paths_list, samples_update_paths_list


