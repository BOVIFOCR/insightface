import sys, os
import cv2
import numpy as np
import torch
import mxnet as mx
from mxnet import ndarray as nd
import copy
import glob
import re


class Loader_3DTEC:
    def __init__(self):
        pass

    
    def load_info_data(self, info_file):
        with open(info_file, 'r') as file:
            all_lines = [line.strip() for line in file.readlines()]
            if all_lines[0].startswith('ShotID'):
                info_keys = all_lines[0].replace('\t\t','\t').split('\t')
                all_lines = all_lines[1:]
            else:
                info_keys = ['ShotID', 'ID', 'S', 'TwinID', 'Emotion', 'TwinLabel']
            
        info_dataset = {}
        for idx_line, line in enumerate(all_lines):
            line_data = line.split('\t')
            assert len(info_keys) == len(line_data), f'Error len(info_keys) ({len(info_keys)}) != len(line_data) ({len(line_data)}), should be equal!'
            
            ShotID = line_data[0]
            info_one_sample = {info_keys[i]:line_data[i] for i in range(1,len(line_data))}
            info_dataset[ShotID] = info_one_sample
        return info_dataset

    
    def load_gallery_or_probe_protocol(self, protocol_file):
        with open(protocol_file, 'r') as file:
            all_lines = [line.strip() for line in file.readlines()]
            return all_lines


    # def load_protocol(self, protocol_file):
    def make_protocol(self, gallery_info, probe_info, info_dataset, only_twins=True):
        if only_twins:
            num_verif_pairs = 0
            for idx_gall, gall_sample_ShotID in enumerate(gallery_info):
                gall_sample_info = info_dataset[gall_sample_ShotID]
                for idx_prob, prob_sample_ShotID in enumerate(probe_info):
                    prob_sample_info = info_dataset[prob_sample_ShotID]
                    if gall_sample_info['TwinID'] == prob_sample_info['TwinID']:
                        num_verif_pairs += 1
        else:
            num_verif_pairs = len(gallery_info) * len(probe_info)


        print('num_verif_pairs:', num_verif_pairs)
        protocol = [None] * num_verif_pairs
        # sys.exit(0)

        pair_id = 0
        for idx_gall, gall_sample_ShotID in enumerate(gallery_info):
            gall_sample_info = info_dataset[gall_sample_ShotID]
            for idx_prob, prob_sample_ShotID in enumerate(probe_info):
                prob_sample_info = info_dataset[prob_sample_ShotID]
                # print('gall_sample_ShotID:', gall_sample_ShotID, gall_sample_info)
                # print('prob_sample_ShotID:', prob_sample_ShotID, prob_sample_info)

                if only_twins:
                    if gall_sample_info['TwinID'] != prob_sample_info['TwinID']:
                        continue

                sample0_subj = gall_sample_info['ID']
                sample0      = gall_sample_ShotID

                sample1_subj = prob_sample_info['ID']
                sample1      = prob_sample_ShotID

                pair_label   = 1 if gall_sample_info['TwinID']    == prob_sample_info['TwinID'] and \
                                    gall_sample_info['TwinLabel'] == prob_sample_info['TwinLabel'] else 0

                pair = {}
                pair['id']           = pair_id

                pair['sample0_subj'] = sample0_subj
                pair['sample0']      = sample0
                
                pair['sample1_subj'] = sample1_subj
                pair['sample1']      = sample1

                pair['pair_label']   = pair_label

                protocol[pair_id] = pair
                pair_id += 1
                # sys.exit(0)

        return protocol


    def natural_sort(self, l):
        convert = lambda text: int(text) if text.isdigit() else text.lower()
        alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
        return sorted(l, key=alphanum_key)


    def update_paths(self, protocol, data_dir, replace_ext='.png', inplace=True):
        if not inplace:
            protocol = copy.deepcopy(protocol)

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
            sample0_pattern = os.path.join(glob.escape(data_dir), sample0_name+'*'+sample0_ext)
            path_sample0 = self.natural_sort(glob.glob(sample0_pattern))
            assert len(path_sample0) > 0, f'Error, no file found with pattern \'{sample0_pattern}\''
            # assert len(path_sample0) < 2, f'Error, more than one file found with pattern \'{sample0_pattern}\': {path_sample0}'
            if len(path_sample0) > 1: num_img_multiple_faces += 1
            path_sample0 = path_sample0[0]

            sample1_name, sample1_ext = os.path.splitext(sample1)
            sample1_pattern = os.path.join(glob.escape(data_dir), sample1_name+'*'+sample1_ext)
            path_sample1 = self.natural_sort(glob.glob(sample1_pattern))
            assert len(path_sample1) > 0, f'Error, no file found with pattern \'{sample1_pattern}\''
            # assert len(path_sample1) < 2, f'Error, more than one file found with pattern \'{sample1_pattern}\': {path_sample1}'
            if len(path_sample1) > 1: num_img_multiple_faces += 1
            path_sample1 = path_sample1[0]

            pair['sample0'] = path_sample0
            pair['sample1'] = path_sample1
            # print('path_sample0:', path_sample0)
            # print('path_sample1:', path_sample1)
            # print('pair_label:', pair_label)
            # print('--------------')
        print('num_img_multiple_faces:', num_img_multiple_faces)
        return protocol


    def load_dataset(self, gallery_file, data_dir, image_size, replace_ext='.png', only_twins=True):
        print(f"Loading gallery file: \'{gallery_file}\'")
        gallery_info = self.load_gallery_or_probe_protocol(gallery_file)
        # print('gallery_info:', gallery_info)
        probe_file = gallery_file.replace('_gallery.','_probe.')
        print(f"Loading probe file: \'{probe_file}\'")
        probe_info = self.load_gallery_or_probe_protocol(probe_file)
        # print('probe_info:', probe_info)
        # sys.exit(0)

        info_dataset_file_name = 'IDTwins-info.txt'
        info_dataset_path = os.path.join(os.path.dirname(gallery_file), info_dataset_file_name)
        assert os.path.isfile(info_dataset_path), f'Error, expected file \'{info_dataset_file_name}\' containing information about the dataset in the same folder of gallery and probe files: \'{os.path.dirname(gallery_file)}\''
        print(f"Loading information dataset file: \'{info_dataset_path}\'")
        info_dataset = self.load_info_data(info_dataset_path)
        # print('info_dataset:', info_dataset)
        # sys.exit(0)

        print(f"Making protocol...")
        pairs_orig   = self.make_protocol(gallery_info, probe_info, info_dataset, only_twins)
        # print('pairs_orig:', pairs_orig)
        pairs_update = self.update_paths(pairs_orig, data_dir, replace_ext, inplace=False)
        # print('len(pairs_orig):', len(pairs_orig))
        # sys.exit(0)

        data_list = []
        for flip in [0, 1]:
            data = torch.empty((len(pairs_update)*2, 3, image_size[0], image_size[1]))
            data_list.append(data)

        issame_list               = np.array([bool(pairs_update[i]['pair_label']) for i in range(len(pairs_update))])
        subj_list                 = np.array([(pairs_update[i]['sample0_subj'], pairs_update[i]['sample1_subj']) for i in range(len(pairs_update))])
        samples_orig_paths_list   = np.array([(pairs_orig[i]['sample0'], pairs_orig[i]['sample1']) for i in range(len(pairs_orig))])
        samples_update_paths_list = np.array([(pairs_update[i]['sample0'], pairs_update[i]['sample1']) for i in range(len(pairs_update))])

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


