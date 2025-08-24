import sys, os
import cv2
import numpy as np
import torch
import mxnet as mx
from mxnet import ndarray as nd
import copy
import glob
import re


class Loader_NDTwins:
    def __init__(self):
        pass


    def load_pairs_table(self, protocol_file):
        with open(protocol_file, 'r') as file:
            all_lines = [line.strip().replace('\"','') for line in file.readlines()]
            # print('all_lines:', all_lines)
            # sys.exit(0)
        cols = all_lines[0].split(',')
        all_lines = all_lines[1:]

        relationships_columns = {col:[] for col in cols}
        for idx_line, line in enumerate(all_lines):
            data = line.split(',')
            for idx_col, col in enumerate(cols):
                relationships_columns[col].append(data[idx_col])
        # print('relationships_columns:', relationships_columns)
        # sys.exit(0)

        # unique_subjs = list(set(relationships_columns['subject_id1'] + relationships_columns['subject_id2']))
        # unique_subjs = self.natural_sort(unique_subjs)
        unique_subjs = []
        unique_relationships = []
        num_skipped_relations = 0
        for idx_line in range(len(all_lines)):
            subject_id1       = relationships_columns['subject_id1'][idx_line]
            subject_id2       = relationships_columns['subject_id2'][idx_line]
            relationship_type = relationships_columns['relationship_type'][idx_line]
            uniq_relation = {'subject_id1':subject_id1, 'subject_id2':subject_id2, 'relationship_type':relationship_type}

            current_list_subject_id1 = relationships_columns['subject_id1'][:idx_line+1]
            current_list_subject_id2 = relationships_columns['subject_id2'][:idx_line+1]
            if len(unique_relationships)==0 or not subject_id1 in current_list_subject_id2:
                # print(idx_line, '- uniq_relation:', uniq_relation, '(Adding)')
                unique_relationships.append(uniq_relation)
            elif subject_id1 in current_list_subject_id2:
                idxs_subj1_in_list2 = np.where(np.array(current_list_subject_id2) == subject_id1)[0]
                subj1_found = False
                for idx_subj1_in_list2 in idxs_subj1_in_list2:
                    if current_list_subject_id1[idx_subj1_in_list2] == subject_id2:
                        subj1_found = True
                        break
                if not subj1_found:
                    # print(idx_line, '- uniq_relation:', uniq_relation, '(Adding)')
                    unique_relationships.append(uniq_relation)
                else:
                    num_skipped_relations += 1

            if not subject_id1 in unique_subjs: unique_subjs.append(subject_id1)
            if not subject_id2 in unique_subjs: unique_subjs.append(subject_id2)

        # print('len(unique_subjs):', len(unique_subjs))
        # print('len(unique_relationships):', len(unique_relationships))
        # print('num_skipped_relations:', num_skipped_relations)
        # sys.exit(0)
        assert len(unique_relationships) + num_skipped_relations == len(all_lines), f"Error, len(unique_relationships)({len(unique_relationships)})+num_skipped_relations({num_skipped_relations}) != len(all_lines)({len(all_lines)}). Should be equal!"
        return unique_subjs, unique_relationships


    def update_subjs_names(self, unique_subjs, pairs_table):
        unique_subjs_update = copy.deepcopy(unique_subjs)
        pairs_table_update  = copy.deepcopy(pairs_table)
        for i in range(len(unique_subjs_update)):
            unique_subjs_update[i] = unique_subjs[i].replace('nd1S','')
        for i in range(len(pairs_table_update)):
            pairs_table_update[i]['subject_id1'] = pairs_table_update[i]['subject_id1'].replace('nd1S','')
            pairs_table_update[i]['subject_id2'] = pairs_table_update[i]['subject_id2'].replace('nd1S','')
        return unique_subjs_update, pairs_table_update


    def make_protocol(self, pairs_table, dict_imgs_paths_by_subj, only_twins=True):
        used_subj = []
        protocol = []
        pair_id = 0
        num_pos_pairs = 0
        num_neg_pairs = 0
        for idx_pair, pair in enumerate(pairs_table):
            subject_id1 = pair['subject_id1']
            subject_id2 = pair['subject_id2']

            # make positive pairs of subject_id1
            if subject_id1 in dict_imgs_paths_by_subj and not subject_id1 in used_subj:
                for i in range(len(dict_imgs_paths_by_subj[subject_id1])-1):
                    for j in range(i+1, len(dict_imgs_paths_by_subj[subject_id1])):
                        sample0_subj = subject_id1
                        sample0 = dict_imgs_paths_by_subj[subject_id1][i]

                        sample1_subj = subject_id1
                        sample1 = dict_imgs_paths_by_subj[subject_id1][j]

                        pair_label = 1   # positive pair

                        pair = {}
                        pair['id']           = pair_id

                        pair['sample0_subj'] = sample0_subj
                        pair['sample0']      = sample0
                        
                        pair['sample1_subj'] = sample1_subj
                        pair['sample1']      = sample1

                        pair['pair_label']   = pair_label

                        protocol.append(pair)
                        num_pos_pairs += 1
                        pair_id += 1

                used_subj.append(subject_id1)

            # make positive pairs of subject_id2
            if subject_id2 in dict_imgs_paths_by_subj and not subject_id2 in used_subj:
                for i in range(len(dict_imgs_paths_by_subj[subject_id2])-1):
                    for j in range(i+1, len(dict_imgs_paths_by_subj[subject_id2])):
                        sample0_subj = subject_id2
                        sample0 = dict_imgs_paths_by_subj[subject_id2][i]

                        sample1_subj = subject_id2
                        sample1 = dict_imgs_paths_by_subj[subject_id2][j]

                        pair_label = 1   # positive pair

                        pair = {}
                        pair['id']           = pair_id

                        pair['sample0_subj'] = sample0_subj
                        pair['sample0']      = sample0
                        
                        pair['sample1_subj'] = sample1_subj
                        pair['sample1']      = sample1

                        pair['pair_label']   = pair_label

                        protocol.append(pair)
                        num_pos_pairs += 1
                        pair_id += 1

                used_subj.append(subject_id2)

            # make negative pairs of subject_id1 and subject_id2
            if subject_id1 in dict_imgs_paths_by_subj and subject_id2 in dict_imgs_paths_by_subj:
                for i in range(len(dict_imgs_paths_by_subj[subject_id1])):
                    for j in range(len(dict_imgs_paths_by_subj[subject_id2])):
                        sample0_subj = subject_id1
                        sample0 = dict_imgs_paths_by_subj[subject_id1][i]

                        sample1_subj = subject_id2
                        sample1 = dict_imgs_paths_by_subj[subject_id2][j]

                        pair_label = 0   # negative pair

                        pair = {}
                        pair['id']           = pair_id

                        pair['sample0_subj'] = sample0_subj
                        pair['sample0']      = sample0
                        
                        pair['sample1_subj'] = sample1_subj
                        pair['sample1']      = sample1

                        pair['pair_label']   = pair_label

                        protocol.append(pair)
                        num_neg_pairs += 1
                        pair_id += 1

        print('num_pos_pairs:', num_pos_pairs)
        print('num_neg_pairs:', num_neg_pairs)
        return protocol


    def natural_sort(self, l):
        convert = lambda text: int(text) if text.isdigit() else text.lower()
        alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
        return sorted(l, key=alphanum_key)


    '''
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
    '''


    def load_dict_imgs_paths_by_subj(self, data_dir, ext=['.jpg', '.jpeg', '.png']):
        img_dict = {}
        for root, dirs, files in os.walk(data_dir):
            subfolder = os.path.basename(root)
            img_files = self.natural_sort([os.path.join(root, f) for f in files if any(f.lower().endswith(e) for e in ext)])
            if img_files:
                if subfolder not in img_dict:
                    img_dict[subfolder] = []
                img_dict[subfolder].extend(img_files)
        return img_dict


    def load_dataset(self, pairs_table_file, data_dir, image_size, replace_ext='.png', only_twins=True):
        print(f"Loading pairs table file: \'{pairs_table_file}\'")
        unique_subjs, pairs_table               = self.load_pairs_table(pairs_table_file)
        # print('unique_subjs:', unique_subjs)
        # print('pairs_table:', pairs_table)
        # sys.exit(0)
        unique_subjs_update, pairs_table_update = self.update_subjs_names(unique_subjs, pairs_table)
        pairs_table_update = pairs_table_update[0:int(len(pairs_table_update)/25.0)]    # PROVISORY (DUE THE WHOLE DATASET DOESN'T FIT ON MERMORY)
        # print('unique_subjs:', unique_subjs)
        # print('pairs_table:', pairs_table)
        # sys.exit(0)

        print(f"Loading imgs paths: \'{data_dir}\'")
        dict_imgs_paths_by_subj = self.load_dict_imgs_paths_by_subj(data_dir)
        # print('dict_imgs_paths_by_subj:', dict_imgs_paths_by_subj)
        # print("dict_imgs_paths_by_subj.keys():", dict_imgs_paths_by_subj.keys())
        # print("dict_imgs_paths_by_subj['90003']:", dict_imgs_paths_by_subj['90003'])
        # sys.exit(0)

        print(f"Making protocol...")
        pairs_orig   = self.make_protocol(pairs_table_update, dict_imgs_paths_by_subj, only_twins)
        # print('pairs_orig:', pairs_orig)
        # pairs_update = self.update_paths(pairs_orig, data_dir, replace_ext, inplace=False)
        pairs_update = pairs_orig
        print('len(pairs_update):', len(pairs_update))
        # sys.exit(0)

        data_list = []
        # num_image_version = 2   # normal and flipped
        num_image_version = 1     # only normal
        for flip in range(num_image_version):
            print(f'    {flip}/{num_image_version} - Allocating images tensor...')
            # data = torch.empty((len(pairs_update)*2, 3, image_size[0], image_size[1]))
            data = torch.zeros((len(pairs_update)*2, 3, image_size[0], image_size[1]))
            data_list.append(data)
        print('    Done')

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
            for flip in range(num_image_version):
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


