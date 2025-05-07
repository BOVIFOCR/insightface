import sys, os
import cv2
import numpy as np
import torch
import mxnet as mx
from mxnet import ndarray as nd
import copy
import glob
import re


class Loader_DoppelVer:
    def __init__(self):
        # wrong annotation labels, images without faces
        self.invalid_subjs = ['Hilary Duff', 'Penelope Cruz', 'Monica Cruz']


    def load_protocol(self, protocol_file):
        with open(protocol_file, 'r') as file:
            all_lines = [line.strip() for line in file.readlines()]
            if all_lines[0].startswith('ID,INDIVIDUAL_1,IMAGE_1,INDIVIDUAL_2,IMAGE_2,LABEL,SPLIT'):
                all_lines = all_lines[1:]
            protocol = len(all_lines) * [None]
            # print('all_lines:', all_lines)

            num_invalid_pairs = 0
            for i, line in enumerate(all_lines):
                # print('line:', line)
                line_data = line.split(',')
                # sys.exit(0)

                pair_id      = line_data[0]

                sample0_subj = line_data[1]
                sample0      = line_data[2]

                sample1_subj = line_data[3]
                sample1      = line_data[4]

                pair_label   = int(line_data[5])
                pair_split   = line_data[6]

                if sample0_subj in self.invalid_subjs or sample1_subj in self.invalid_subjs:
                    num_invalid_pairs += 1
                    continue   # wrong annotation labels, images without faces

                pair = {}
                pair['id']           = pair_id
                pair['sample0_subj'] = sample0_subj
                pair['sample0']      = sample0
                
                pair['sample1_subj'] = sample1_subj
                pair['sample1']      = sample1

                pair['pair_label']   = pair_label
                pair['pair_split']   = pair_split

                protocol[i] = pair
            
            print('num_invalid_pairs:', num_invalid_pairs, '    invalid_subjs:', self.invalid_subjs)
            protocol = list(filter((None).__ne__, protocol))  # remove all None items
            return protocol


    def natural_sort(self, l):
        convert = lambda text: int(text) if text.isdigit() else text.lower()
        alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
        return sorted(l, key=alphanum_key)


    def remove_elements_by_indices(self, data_list, indices_to_remove):
        if not indices_to_remove:
            return list(data_list)
        indices_to_remove = sorted(indices_to_remove, reverse=True)
        new_list = list(data_list)
        for index in indices_to_remove:
            del new_list[index]
        return new_list


    def update_paths(self, protocol, data_dir, replace_ext='.png', ignore_missing_imgs=False, inplace=True):
        if not inplace:
            protocol = copy.deepcopy(protocol)

        indexes_invalid_pairs = []
        num_img_multiple_faces = 0
        for i, pair in enumerate(protocol):
            sample0 = pair['sample0']
            sample1 = pair['sample1']
            # print('sample0:', sample0)
            # print('sample1:', sample1)

            if replace_ext != '':
                curr_ext = sample0.split('.')[-1]
                sample0 = sample0.replace('.'+curr_ext, replace_ext)
                sample1 = sample1.replace('.'+curr_ext, replace_ext)
                # print('sample0:', sample0)
                # print('sample1:', sample1)
            
            sample0_name, sample0_ext = os.path.splitext(sample0)
            sample0_pattern = os.path.join(glob.escape(os.path.join(data_dir, pair['sample0_subj'])), sample0_name+'*'+sample0_ext)
            path_sample0 = self.natural_sort(glob.glob(sample0_pattern))
            if not ignore_missing_imgs:
                assert len(path_sample0) > 0, f'Error, no file found with pattern \'{sample0_pattern}\''
                # assert len(path_sample0) < 2, f'Error, more than one file found with pattern \'{sample0_pattern}\': {path_sample0}'
                if len(path_sample0) > 1: num_img_multiple_faces += 1
                path_sample0 = path_sample0[0]
            else:
                if len(path_sample0) == 0 and not i in indexes_invalid_pairs:
                    indexes_invalid_pairs.append(i)
                elif len(path_sample0) > 0:
                    path_sample0 = path_sample0[0]

            sample1_name, sample1_ext = os.path.splitext(sample1)
            sample1_pattern = os.path.join(glob.escape(os.path.join(data_dir, pair['sample1_subj'])), sample1_name+'*'+sample1_ext)
            path_sample1 = self.natural_sort(glob.glob(sample1_pattern))
            if not ignore_missing_imgs:
                assert len(path_sample1) > 0, f'Error, no file found with pattern \'{sample1_pattern}\''
                # assert len(path_sample1) < 2, f'Error, more than one file found with pattern \'{sample1_pattern}\': {path_sample1}'
                if len(path_sample1) > 1: num_img_multiple_faces += 1
                path_sample1 = path_sample1[0]
            else:
                if len(path_sample1) == 0 and not i in indexes_invalid_pairs:
                    indexes_invalid_pairs.append(i)
                elif len(path_sample1) > 0:
                    path_sample1 = path_sample1[0]

            pair['sample0'] = path_sample0
            pair['sample1'] = path_sample1
            # print('path_sample0:', path_sample0)
            # print('path_sample1:', path_sample1)
            # print('pair_label:', pair_label)
            # print('--------------')
        print('num_img_multiple_faces:', num_img_multiple_faces)
        print('num removed pairs due missing imgs:', len(indexes_invalid_pairs))
        
        indexes_invalid_pairs = list(set(indexes_invalid_pairs))
        if len(indexes_invalid_pairs) > 0:
            protocol = self.remove_elements_by_indices(protocol, indexes_invalid_pairs)

        return protocol


    def load_dataset(self, protocol_file, data_dir, image_size, ignore_missing_imgs=False, replace_ext='.png'):
        print(f"Loading protocol: \'{protocol_file}\'")
        pairs_orig = self.load_protocol(protocol_file)
        pairs_update = self.update_paths(pairs_orig, data_dir, replace_ext, ignore_missing_imgs, inplace=False)

        data_list = []
        for flip in [0, 1]:
            data = torch.empty((len(pairs_update)*2, 3, image_size[0], image_size[1]))
            data_list.append(data)

        issame_list               = np.array([bool(pairs_update[i]['pair_label']) for i in range(len(pairs_update))])
        # races_list              = np.array([sorted((pairs_update[i]['sample0_race'], pairs_update[i]['sample1_race'])) for i in range(len(pairs_update))])
        subj_list                 = np.array([(pairs_update[i]['sample0_subj'], pairs_update[i]['sample1_subj']) for i in range(len(pairs_update))])
        samples_orig_paths_list   = np.array([(pairs_orig[i]['sample0'], pairs_orig[i]['sample1']) for i in range(len(pairs_orig))])
        samples_update_paths_list = np.array([(pairs_update[i]['sample0'], pairs_update[i]['sample1']) for i in range(len(pairs_update))])
        # for i, (label, races, subjs) in enumerate(zip(issame_list, races_list, subj_list)):
            # print(f'pair:{i} - label: {label} - races: {races} - subjs: {subjs}')
            # if races[0] == races[1]:
            #     print(f'pair:{i} - label: {label} - races: {races} - subjs: {subjs}')
        # print('len(issame_list):', len(issame_list))
        # print('len(races_list):', len(races_list))
        # print('len(subj_list):', len(subj_list))
        # sys.exit(0)

        for idx in range(len(pairs_update) * 2):
            # _bin = bins[idx]
            # img = mx.image.imdecode(_bin)
            idx_pair = int(idx/2)
            if idx % 2 == 0:
                img_path = pairs_update[idx_pair]['sample0']
                # img = cv2.imread(pairs_update[idx_pair]['sample0'])
            else:
                img_path = pairs_update[idx_pair]['sample1']
                # img = cv2.imread(pairs_update[idx_pair]['sample1'])
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
        # return data_list, issame_list, subj_list, samples_orig_paths_list, samples_update_paths_list
        return {'data_list': data_list,
                'issame_list': issame_list,
                'subj_list': subj_list, 
                'samples_orig_paths_list': samples_orig_paths_list, 
                'samples_update_paths_list': samples_update_paths_list}


