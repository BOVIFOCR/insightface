# duo
# export CUDA_VISIBLE_DEVICES=0; python verification_TCDiff.py --data-dir /datasets2/1st_frcsyn_wacv2024/datasets/real/3_BUPT-BalancedFace/race_per_7000_crops_112x112 --network r100 --model ../work_dirs/casia_frcsyn_r100/2023-10-14_09-51-11_GPU0/model.pt --target bupt --protocol /datasets2/1st_frcsyn_wacv2024/comparison_files/comparison_files/sub-tasks_1.1_1.2/bupt_comparison.txt --style-clusters-data /datasets2/1st_frcsyn_wacv2024/datasets/real/3_BUPT-BalancedFace/race_per_7000_crops_112x112_JUST-PROTOCOL-IMGS_STYLE_FEATURES_CLUSTERING_FROM_1_CASIA-WebFace-imgs_crops_112x112_STYLE_FEATURES_CLUSTERING-feature=_style-_distance=cosine-nclusters=100/feature=_style/_distance=cosine/nclusters=100/clusters-data_feature=_style.pt_distance=cosine_nclusters=100.pkl

# dataset HDA-Doppelganger
# export CUDA_VISIBLE_DEVICES=0; python verification_TCDiff.py --network r100 --model ../work_dirs/casia_frcsyn_r100/2023-10-14_09-51-11_GPU0/model.pt --target hda_doppelganger --data-dir /datasets1/bjgbiesseck/doppelgangers_lookalikes/HDA-Doppelgaenger_DETECTED_FACES_RETINAFACE_scales=[1.0,0.5,0.25]_nms=0.4/imgs

# dataset DoppelVer
# export CUDA_VISIBLE_DEVICES=0; python verification_TCDiff.py --network r100 --model ../work_dirs/casia_frcsyn_r100/2023-10-14_09-51-11_GPU0/model.pt --target doppelver_doppelganger --data-dir /datasets1/bjgbiesseck/doppelgangers_lookalikes/DoppelVer/Images/CCA_Images_DETECTED_FACES_RETINAFACE_scales=[1.0,0.5,0.25]_nms=0.4/imgs --protocol /datasets1/bjgbiesseck/doppelgangers_lookalikes/DoppelVer/DoppelgangerProtocol.csv
# export CUDA_VISIBLE_DEVICES=0; python verification_TCDiff.py --network r100 --model ../work_dirs/casia_frcsyn_r100/2023-10-14_09-51-11_GPU0/model.pt --target doppelver_vise --data-dir /datasets1/bjgbiesseck/doppelgangers_lookalikes/DoppelVer/Images/CCA_Images_DETECTED_FACES_RETINAFACE_scales=[1.0,0.5,0.25]_nms=0.4/imgs --protocol /datasets1/bjgbiesseck/doppelgangers_lookalikes/DoppelVer/ViSEProtocol.csv

# dataset 3D-TEC
# export CUDA_VISIBLE_DEVICES=0; python verification_TCDiff.py --network r100 --model ../work_dirs/casia_frcsyn_r100/2023-10-14_09-51-11_GPU0/model.pt --target 3d_tec --data-dir /datasets1/bjgbiesseck/3D-Twins-Expression-Challenge_3D-TEC_DETECTED_FACES_RETINAFACE_scales=[0.5]_nms=0.4/imgs/textureimages --protocol /datasets1/bjgbiesseck/3D-Twins-Expression-Challenge_3D-TEC/exp1_gallery.txt
# export CUDA_VISIBLE_DEVICES=0; python verification_TCDiff.py --network r100 --model ../work_dirs/casia_frcsyn_r100/2023-10-14_09-51-11_GPU0/model.pt --target 3d_tec --data-dir /datasets1/bjgbiesseck/3D-Twins-Expression-Challenge_3D-TEC_DETECTED_FACES_RETINAFACE_scales=[0.5]_nms=0.4/imgs/textureimages --protocol /datasets1/bjgbiesseck/3D-Twins-Expression-Challenge_3D-TEC/exp3_gallery.txt

# dataset ND-Twins-2009-2010
# export CUDA_VISIBLE_DEVICES=0; python verification_TCDiff.py --network r100 --model ../work_dirs/casia_frcsyn_r100/2023-10-14_09-51-11_GPU0/model.pt --target nd_twins --data-dir /datasets1/bjgbiesseck/ND-Twins-2009-2010/images_DETECTED_FACES_RETINAFACE_scales=[0.25]_nms=0.4/imgs --protocol /datasets1/bjgbiesseck/ND-Twins-2009-2010/TwinsChallenge_1.0.0/TwinsChallenge/data/TwinsPairTable.csv

"""Helper for evaluation on the Labeled Faces in the Wild dataset 
"""

# MIT License
#
# Copyright (c) 2016 David Sandberg
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import datetime
import os, sys
import pickle

import torch
import mxnet as mx
import numpy as np
import sklearn
import torch
from mxnet import ndarray as nd
from scipy import interpolate
from scipy.stats import entropy
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from PIL import Image
import matplotlib.pyplot as plt

sys.path.insert(0, "../")
from backbones import get_model

import argparse   # Bernardo
import itertools

from loader_BUPT import Loader_BUPT
from loader_HDA_Doppelganger import Loader_HDA_Doppelganger
from loader_DoppelVer import Loader_DoppelVer
from loader_3DTEC import Loader_3DTEC
from loader_NDTwins import Loader_NDTwins



def load_dict(path: str) -> dict:
    with open(path, 'rb') as file:
        return pickle.load(file)


def save_img(pathfile, img):
    img = img.squeeze(0).permute(1, 2, 0).byte().numpy()
    pil_img = Image.fromarray(img)
    pil_img.save(pathfile, format="PNG")


class LFold:
    def __init__(self, n_splits=2, shuffle=False):
        self.n_splits = n_splits
        if self.n_splits > 1:
            self.k_fold = KFold(n_splits=n_splits, shuffle=shuffle)

    def split(self, indices):
        if self.n_splits > 1:
            return self.k_fold.split(indices)
        else:
            return [(indices, indices)]


def calculate_roc(thresholds,
                  embeddings1,
                  embeddings2,
                  actual_issame,
                  nrof_folds=10,
                  pca=0):
    assert (embeddings1.shape[0] == embeddings2.shape[0])
    assert (embeddings1.shape[1] == embeddings2.shape[1])
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = LFold(n_splits=nrof_folds, shuffle=False)

    tprs = np.zeros((nrof_folds, nrof_thresholds))
    fprs = np.zeros((nrof_folds, nrof_thresholds))
    accuracy = np.zeros((nrof_folds))
    indices = np.arange(nrof_pairs)

    if pca == 0:
        diff = np.subtract(embeddings1, embeddings2)
        dist = np.sum(np.square(diff), 1)

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        if pca > 0:
            print('doing pca on', fold_idx)
            embed1_train = embeddings1[train_set]
            embed2_train = embeddings2[train_set]
            _embed_train = np.concatenate((embed1_train, embed2_train), axis=0)
            pca_model = PCA(n_components=pca)
            pca_model.fit(_embed_train)
            embed1 = pca_model.transform(embeddings1)
            embed2 = pca_model.transform(embeddings2)
            embed1 = sklearn.preprocessing.normalize(embed1)
            embed2 = sklearn.preprocessing.normalize(embed2)
            diff = np.subtract(embed1, embed2)
            dist = np.sum(np.square(diff), 1)

        # Find the best threshold for the fold
        acc_train = np.zeros((nrof_thresholds))
        for threshold_idx, threshold in enumerate(thresholds):
            _, _, acc_train[threshold_idx] = calculate_accuracy(
                threshold, dist[train_set], actual_issame[train_set])
        best_threshold_index = np.argmax(acc_train)
        for threshold_idx, threshold in enumerate(thresholds):
            tprs[fold_idx, threshold_idx], fprs[fold_idx, threshold_idx], _ = calculate_accuracy(
                threshold, dist[test_set],
                actual_issame[test_set])
        _, _, accuracy[fold_idx] = calculate_accuracy(
            thresholds[best_threshold_index], dist[test_set],
            actual_issame[test_set])

    tpr = np.mean(tprs, 0)
    fpr = np.mean(fprs, 0)
    return tpr, fpr, accuracy


def calculate_accuracy(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(
        np.logical_and(np.logical_not(predict_issame),
                       np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))

    tpr = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)
    fpr = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)
    acc = float(tp + tn) / dist.size
    return tpr, fpr, acc


def calculate_val(thresholds,
                  embeddings1,
                  embeddings2,
                  actual_issame,
                  far_target,
                  nrof_folds=10):
    assert (embeddings1.shape[0] == embeddings2.shape[0])
    assert (embeddings1.shape[1] == embeddings2.shape[1])
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = LFold(n_splits=nrof_folds, shuffle=False)

    val = np.zeros(nrof_folds)
    far = np.zeros(nrof_folds)

    diff = np.subtract(embeddings1, embeddings2)
    dist = np.sum(np.square(diff), 1)
    indices = np.arange(nrof_pairs)

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):

        # Find the threshold that gives FAR = far_target
        far_train = np.zeros(nrof_thresholds)
        for threshold_idx, threshold in enumerate(thresholds):
            _, far_train[threshold_idx] = calculate_val_far(
                threshold, dist[train_set], actual_issame[train_set])
        if np.max(far_train) >= far_target:
            f = interpolate.interp1d(far_train, thresholds, kind='slinear')
            threshold = f(far_target)
        else:
            threshold = 0.0

        val[fold_idx], far[fold_idx] = calculate_val_far(
            threshold, dist[test_set], actual_issame[test_set])

    val_mean = np.mean(val)
    far_mean = np.mean(far)
    val_std = np.std(val)
    return val_mean, val_std, far_mean


def calculate_val_far(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    true_accept = np.sum(np.logical_and(predict_issame, actual_issame))
    false_accept = np.sum(
        np.logical_and(predict_issame, np.logical_not(actual_issame)))
    n_same = np.sum(actual_issame)
    n_diff = np.sum(np.logical_not(actual_issame))
    # print(true_accept, false_accept)
    # print(n_same, n_diff)
    val = float(true_accept) / float(n_same)
    far = float(false_accept) / float(n_diff)
    return val, far


def evaluate(embeddings, actual_issame, nrof_folds=10, pca=0):
    # Calculate evaluation metrics
    thresholds = np.arange(0, 4, 0.01)
    embeddings1 = embeddings[0::2]
    embeddings2 = embeddings[1::2]
    tpr, fpr, accuracy = calculate_roc(thresholds,
                                       embeddings1,
                                       embeddings2,
                                       np.asarray(actual_issame),
                                       nrof_folds=nrof_folds,
                                       pca=pca)
    thresholds = np.arange(0, 4, 0.001)
    val, val_std, far = calculate_val(thresholds,
                                      embeddings1,
                                      embeddings2,
                                      np.asarray(actual_issame),
                                      1e-3,
                                      nrof_folds=nrof_folds)
    return tpr, fpr, accuracy, val, val_std, far


@torch.no_grad()
def load_bin(path, image_size):
    try:
        with open(path, 'rb') as f:
            bins, issame_list = pickle.load(f)  # py2
    except UnicodeDecodeError as e:
        with open(path, 'rb') as f:
            bins, issame_list = pickle.load(f, encoding='bytes')  # py3
    data_list = []
    for flip in [0, 1]:
        data = torch.empty((len(issame_list) * 2, 3, image_size[0], image_size[1]))
        data_list.append(data)
    for idx in range(len(issame_list) * 2):
        _bin = bins[idx]
        img = mx.image.imdecode(_bin)
        if img.shape[1] != image_size[0]:
            img = mx.image.resize_short(img, image_size[0])
        img = nd.transpose(img, axes=(2, 0, 1))
        for flip in [0, 1]:
            if flip == 1:
                img = mx.ndarray.flip(data=img, axis=2)
            data_list[flip][idx][:] = torch.from_numpy(img.asnumpy())
        if idx % 1000 == 0:
            print('loading bin', idx)
    print(data_list[0].shape)
    return data_list, issame_list


@torch.no_grad()
def test(data_set, backbone, batch_size, nfolds=10):
    data_list = data_set[0]
    issame_list = data_set[1]
    embeddings_list = []
    time_consumed = 0.0
    for i in range(len(data_list)):
        data = data_list[i]
        embeddings = None
        ba = 0
        while ba < data.shape[0]:
            bb = min(ba + batch_size, data.shape[0])
            count = bb - ba
            _data = data[bb - batch_size: bb]
            time0 = datetime.datetime.now()
            img = ((_data / 255) - 0.5) / 0.5
            # print('img:', img)
            # print('img.size():', img.size())

            net_out: torch.Tensor = backbone(img)         # original
            # net_out: torch.Tensor = backbone.forward(img)   # Bernardo

            _embeddings = net_out.detach().cpu().numpy()
            time_now = datetime.datetime.now()
            diff = time_now - time0
            time_consumed += diff.total_seconds()
            if embeddings is None:
                embeddings = np.zeros((data.shape[0], _embeddings.shape[1]))
            embeddings[ba:bb, :] = _embeddings[(batch_size - count):, :]
            ba = bb
        embeddings_list.append(embeddings)

    _xnorm = 0.0
    _xnorm_cnt = 0
    for embed in embeddings_list:
        for i in range(embed.shape[0]):
            _em = embed[i]
            _norm = np.linalg.norm(_em)
            _xnorm += _norm
            _xnorm_cnt += 1
    _xnorm /= _xnorm_cnt

    embeddings = embeddings_list[0].copy()
    embeddings = sklearn.preprocessing.normalize(embeddings)
    acc1 = 0.0
    std1 = 0.0
    embeddings = embeddings_list[0] + embeddings_list[1]
    embeddings = sklearn.preprocessing.normalize(embeddings)
    print(embeddings.shape)
    print('infer time', time_consumed)
    _, _, accuracy, val, val_std, far = evaluate(embeddings, issame_list, nrof_folds=nfolds)
    acc2, std2 = np.mean(accuracy), np.std(accuracy)
    return acc1, std1, acc2, std2, _xnorm, embeddings_list







###################################################
# RACES ANALYSIS (African, Asian, Caucasian, Indian)
###################################################

def cosine_sim(embeddings1, embeddings2):
    sims = np.zeros(embeddings1.shape[0])
    for i in range(0,embeddings1.shape[0]):
        sims[i] = float(np.maximum(np.dot(embeddings1[i],embeddings2[i])/(np.linalg.norm(embeddings1[i])*np.linalg.norm(embeddings2[i])), 0.0))
    return sims


def cosine_dist(embeddings1, embeddings2):
    distances = 1. - cosine_sim(embeddings1, embeddings2)
    return distances


def compute_score(embeddings1, embeddings2, score):
    if score == 'eucl-dist':
        diff = np.subtract(embeddings1, embeddings2)
        dist = np.sum(np.square(diff), 1)
    elif score == 'cos-dist':
        dist = cosine_dist(embeddings1, embeddings2)
    elif score == 'cos-sim':
        dist = cosine_sim(embeddings1, embeddings2)
    return dist


def get_predict_true(dist, threshold, score):
    if score == 'eucl-dist' or score == 'cos-dist':
        predict_issame = np.less(dist, threshold)
    elif score == 'cos-sim':
        predict_issame = np.greater_equal(dist, threshold)
    return predict_issame


def fuse_scores(score1, score2):
    # score1 = (score1 - score1.min()) / (score1.max() - score1.min())
    # score2 = (score2 - score2.min()) / (score2.max() - score2.min())
    # score1 = score1 / score1.max()
    # score2 = score2 / score2.max()

    fused = (score1 + score2) / 2.0
    return fused


def get_races_combinations():
    # races = ['African', 'Asian', 'Caucasian', 'Indian']
    races = ['Asian', 'Indian', 'African', 'Caucasian']
    races_comb = [(race, race) for race in races]
    # races_comb += list(itertools.combinations(races, 2))
    # return sorted(races_comb)
    return races_comb


def get_avg_roc_metrics_races(metrics_races=[{}], races_combs=[]):
    avg_roc_metrics = {}
    for i, race_comb in enumerate(races_combs):
        accs = [metrics_races[fold_idx][race_comb]['acc'] for fold_idx in range(len(metrics_races))]
        tprs = [metrics_races[fold_idx][race_comb]['tpr'] for fold_idx in range(len(metrics_races))]
        fprs = [metrics_races[fold_idx][race_comb]['fpr'] for fold_idx in range(len(metrics_races))]

        avg_roc_metrics[race_comb] = {}
        avg_roc_metrics[race_comb]['acc_mean'] = np.mean(accs)
        avg_roc_metrics[race_comb]['acc_std']  = np.std(accs)
        avg_roc_metrics[race_comb]['tpr_mean'] = np.mean(tprs)
        avg_roc_metrics[race_comb]['tpr_std']  = np.std(tprs)
        avg_roc_metrics[race_comb]['fpr_mean'] = np.mean(fprs)
        avg_roc_metrics[race_comb]['fpr_std']  = np.std(fprs)

        if 'acc_clusters' in metrics_races[0][races_combs[0]].keys():
            accs_clusters = [metrics_races[fold_idx][race_comb]['acc_clusters'] for fold_idx in range(len(metrics_races))]
            avg_roc_metrics[race_comb]['acc_clusters_mean'] = np.mean(np.stack(accs_clusters), axis=0)
            avg_roc_metrics[race_comb]['acc_clusters_std']  = np.std(np.stack(accs_clusters), axis=0)
            
            perc_hits_same_style_clusters = [metrics_races[fold_idx][race_comb]['perc_hits_same_style_clusters'] for fold_idx in range(len(metrics_races))]
            avg_roc_metrics[race_comb]['perc_hits_same_style_clusters_mean'] = np.mean(np.stack(perc_hits_same_style_clusters), axis=0)
            avg_roc_metrics[race_comb]['perc_hits_same_style_clusters_std']  = np.std(np.stack(perc_hits_same_style_clusters), axis=0)
            
            perc_hits_diff_style_clusters = [metrics_races[fold_idx][race_comb]['perc_hits_diff_style_clusters'] for fold_idx in range(len(metrics_races))]
            avg_roc_metrics[race_comb]['perc_hits_diff_style_clusters_mean'] = np.mean(np.stack(perc_hits_diff_style_clusters), axis=0)
            avg_roc_metrics[race_comb]['perc_hits_diff_style_clusters_std']  = np.std(np.stack(perc_hits_diff_style_clusters), axis=0)
        
        # print(f"avg_roc_metrics[{race_comb}]['acc_clusters_mean']:", avg_roc_metrics[race_comb]['acc_clusters_mean'])
        # sys.exit(0)

    if 'acc_clusters' in metrics_races[0][races_combs[0]].keys():
        avg_roc_metrics['total_races'] = {}
        avg_roc_metrics['total_races']['acc_clusters_mean'] = np.zeros_like(metrics_races[0][races_combs[0]]['acc_clusters'])
        avg_roc_metrics['total_races']['perc_hits_same_style_clusters_mean'] = np.zeros_like(metrics_races[0][races_combs[0]]['acc_clusters'])
        avg_roc_metrics['total_races']['perc_hits_diff_style_clusters_mean'] = np.zeros_like(metrics_races[0][races_combs[0]]['acc_clusters'])
        for i, race_comb in enumerate(races_combs):
            avg_roc_metrics['total_races']['acc_clusters_mean'] += avg_roc_metrics[race_comb]['acc_clusters_mean']
            avg_roc_metrics['total_races']['perc_hits_same_style_clusters_mean'] += avg_roc_metrics[race_comb]['perc_hits_same_style_clusters_mean']
            avg_roc_metrics['total_races']['perc_hits_diff_style_clusters_mean'] += avg_roc_metrics[race_comb]['perc_hits_diff_style_clusters_mean']

        avg_roc_metrics['total_races']['acc_clusters_mean'] /= len(races_combs)
        avg_roc_metrics['total_races']['perc_hits_same_style_clusters_mean'] /= len(races_combs)
        avg_roc_metrics['total_races']['perc_hits_diff_style_clusters_mean'] /= len(races_combs)
    
    return avg_roc_metrics


def get_avg_val_metrics_races(metrics_races=[{}], races_combs=[]):
    avg_val_metrics = {}
    for i, race_comb in enumerate(races_combs):
        vals = [metrics_races[fold_idx][race_comb]['val'] for fold_idx in range(len(metrics_races))]
        fars = [metrics_races[fold_idx][race_comb]['far'] for fold_idx in range(len(metrics_races))]

        avg_val_metrics[race_comb] = {}
        avg_val_metrics[race_comb]['val_mean'] = np.mean(vals)
        avg_val_metrics[race_comb]['val_std']  = np.std(vals)
        avg_val_metrics[race_comb]['far_mean'] = np.mean(fars)
        avg_val_metrics[race_comb]['far_std']  = np.std(fars)
    return avg_val_metrics



def calculate_roc_analyze_races(args, thresholds,
                  embeddings1,
                  embeddings2,
                  actual_issame,
                  races_list,
                  subj_list,
                  nrof_folds=10,
                  pca=0,
                  races_combs=[],
                  style_clusters_data={}):
    assert (embeddings1.shape[0] == embeddings2.shape[0])
    assert (embeddings1.shape[1] == embeddings2.shape[1])
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = LFold(n_splits=nrof_folds, shuffle=False)

    tprs = np.zeros((nrof_folds, nrof_thresholds))
    fprs = np.zeros((nrof_folds, nrof_thresholds))
    accuracy = np.zeros((nrof_folds))
    indices = np.arange(nrof_pairs)
    metrics_races = [None] * nrof_folds
    metrics_style_clusters = [None] * nrof_folds

    if pca == 0:
        # diff = np.subtract(embeddings1, embeddings2)
        # dist = np.sum(np.square(diff), 1)
        # dist = cosine_dist(embeddings1, embeddings2)
        dist = compute_score(embeddings1, embeddings2, args.score)
        

    # Bernardo
    dist_fusion = None
    if args.fusion_dist != '':
        print(f'Loading dist for fusion: \'{args.fusion_dist}\'...')
        dist_fusion = np.load(args.fusion_dist)
        print(f'Fusing scores...\n')
        assert dist.shape[0] == dist_fusion.shape[0]
        dist = fuse_scores(dist, dist_fusion)

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        if pca > 0:
            print('doing pca on', fold_idx)
            embed1_train = embeddings1[train_set]
            embed2_train = embeddings2[train_set]
            _embed_train = np.concatenate((embed1_train, embed2_train), axis=0)
            pca_model = PCA(n_components=pca)
            pca_model.fit(_embed_train)
            embed1 = pca_model.transform(embeddings1)
            embed2 = pca_model.transform(embeddings2)
            embed1 = sklearn.preprocessing.normalize(embed1)
            embed2 = sklearn.preprocessing.normalize(embed2)
            # diff = np.subtract(embed1, embed2)
            # dist = np.sum(np.square(diff), 1)
            # dist = cosine_dist(embed1, embed2)
            dist = compute_score(embed1, embed1, args.score)

            if not dist_fusion is None:
                print(f'Fusing scores (pca)...')
                assert dist.shape[0] == dist_fusion.shape[0]
                dist = fuse_scores(dist, dist_fusion)

        # Find the best threshold for the fold
        acc_train = np.zeros((nrof_thresholds))
        for threshold_idx, threshold in enumerate(thresholds):
            _, _, acc_train[threshold_idx], _ = calculate_accuracy_analyze_races(
                args, threshold, dist[train_set], actual_issame[train_set], races_list=None, subj_list=None, races_combs=None, style_clusters_data=None)
        best_threshold_index = np.argmax(acc_train)
        for threshold_idx, threshold in enumerate(thresholds):
            tprs[fold_idx, threshold_idx], fprs[fold_idx, threshold_idx], _, _ = calculate_accuracy_analyze_races(
                args, threshold, dist[test_set], actual_issame[test_set], races_list=None, subj_list=None, races_combs=None, style_clusters_data=None)
        
        if not races_list is None and not subj_list is None:
            _, _, accuracy[fold_idx], metrics_races[fold_idx], _ = calculate_accuracy_analyze_races(
                args, thresholds[best_threshold_index], dist[test_set],
                actual_issame[test_set], races_list[test_set], subj_list[test_set], races_combs=races_combs, style_clusters_data=None)

            if not style_clusters_data is None:
                _, _, accuracy[fold_idx], metrics_races[fold_idx], _ = calculate_accuracy_analyze_races(
                    args, thresholds[best_threshold_index], dist[test_set],
                    actual_issame[test_set], races_list[test_set], subj_list[test_set], races_combs=races_combs, style_clusters_data=style_clusters_data)

        else:
            _, _, accuracy[fold_idx], _ = calculate_accuracy_analyze_races(
                args, thresholds[best_threshold_index], dist[test_set],
                actual_issame[test_set], races_list=None, subj_list=None, races_combs=races_combs, style_clusters_data=None)

    avg_roc_metrics = None
    if not races_list is None and not subj_list is None:
        avg_roc_metrics = get_avg_roc_metrics_races(metrics_races, races_combs)
    
    tpr = np.mean(tprs, 0)
    fpr = np.mean(fprs, 0)
    return tpr, fpr, accuracy, avg_roc_metrics


def calculate_accuracy_analyze_races(args, threshold, dist, actual_issame, races_list, subj_list, races_combs, style_clusters_data):
    # predict_issame = np.less(dist, threshold)
    predict_issame = get_predict_true(dist, threshold, args.score)

    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))

    tpr = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)
    fpr = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)
    acc = float(tp + tn) / dist.size

    if not style_clusters_data is None:
        nclusters = len(style_clusters_data['cluster_centers_tsne'])
        style_clusters_pairs_labels = style_clusters_data['pairs_cluster_ids']

    # race analysis (African, Asian, Caucasian, Indian)
    if not races_list is None and not subj_list is None:
        metrics_races = {}
        for race_comb in races_combs:
            metrics_races[race_comb] = {}

        for i, race_comb in enumerate(races_combs):
            indices_race_comb = np.where(np.all(races_list == race_comb, axis=1))[0]

            logical_tp = np.logical_and(predict_issame[indices_race_comb], actual_issame[indices_race_comb])
            logical_fp = np.logical_and(predict_issame[indices_race_comb], np.logical_not(actual_issame[indices_race_comb]))
            logical_tn = np.logical_and(np.logical_not(predict_issame[indices_race_comb]), np.logical_not(actual_issame[indices_race_comb]))
            logical_fn = np.logical_and(np.logical_not(predict_issame[indices_race_comb]), actual_issame[indices_race_comb])

            metrics_races[race_comb]['tp'] = np.sum(logical_tp)
            metrics_races[race_comb]['fp'] = np.sum(logical_fp)
            metrics_races[race_comb]['tn'] = np.sum(logical_tn)
            metrics_races[race_comb]['fn'] = np.sum(logical_fn)

            metrics_races[race_comb]['tpr'] = 0 if (metrics_races[race_comb]['tp'] + metrics_races[race_comb]['fn'] == 0) else float(metrics_races[race_comb]['tp']) / float(metrics_races[race_comb]['tp'] + metrics_races[race_comb]['fn'])
            metrics_races[race_comb]['fpr'] = 0 if (metrics_races[race_comb]['fp'] + metrics_races[race_comb]['tn'] == 0) else float(metrics_races[race_comb]['fp']) / float(metrics_races[race_comb]['fp'] + metrics_races[race_comb]['tn'])
            metrics_races[race_comb]['acc'] = 0 if indices_race_comb.size == 0 else float(metrics_races[race_comb]['tp'] + metrics_races[race_comb]['tn']) / indices_race_comb.size
            # print('indices_race_comb:', indices_race_comb, '    type(indices_race_comb):', type(indices_race_comb))
            # sys.exit(0)

            # style faces clusters analysis
            if not style_clusters_data is None:
                metrics_races[race_comb]['tp_clusters']                   = np.zeros((nclusters,))
                metrics_races[race_comb]['fp_clusters']                   = np.zeros((nclusters,))
                metrics_races[race_comb]['tn_clusters']                   = np.zeros((nclusters,))
                metrics_races[race_comb]['fn_clusters']                   = np.zeros((nclusters,))
                metrics_races[race_comb]['acc_clusters']                  = np.zeros((nclusters,))
                metrics_races[race_comb]['num_samples_clusters']          = np.zeros((nclusters,))
                # metrics_races[race_comb]['num_pairs_same_style_clusters'] = np.zeros((nclusters,))
                metrics_races[race_comb]['num_hits_same_style_clusters']  = np.zeros((nclusters,))
                metrics_races[race_comb]['num_hits_diff_style_clusters']  = np.zeros((nclusters,))
                metrics_races[race_comb]['perc_hits_same_style_clusters'] = np.zeros((nclusters,))
                metrics_races[race_comb]['perc_hits_diff_style_clusters'] = np.zeros((nclusters,))

                cluster_pairs_labels_race_sample0 = style_clusters_pairs_labels[indices_race_comb][:,0]
                cluster_pairs_labels_race_sample1 = style_clusters_pairs_labels[indices_race_comb][:,1]
                # same_style_cluster_pairs_labels_race = cluster_pairs_labels_race_sample0 == cluster_pairs_labels_race_sample1

                indices_tp_race = np.where(logical_tp == True)[0]
                np.add.at(metrics_races[race_comb]['tp_clusters'], cluster_pairs_labels_race_sample0[indices_tp_race], 1)
                np.add.at(metrics_races[race_comb]['tp_clusters'], cluster_pairs_labels_race_sample1[indices_tp_race], 1)

                indices_fp_race = np.where(logical_fp == True)[0]
                np.add.at(metrics_races[race_comb]['fp_clusters'], cluster_pairs_labels_race_sample0[indices_fp_race], 1)
                np.add.at(metrics_races[race_comb]['fp_clusters'], cluster_pairs_labels_race_sample1[indices_fp_race], 1)

                indices_tn_race = np.where(logical_tn == True)[0]
                np.add.at(metrics_races[race_comb]['tn_clusters'], cluster_pairs_labels_race_sample0[indices_tn_race], 1)
                np.add.at(metrics_races[race_comb]['tn_clusters'], cluster_pairs_labels_race_sample1[indices_tn_race], 1)

                indices_fn_race = np.where(logical_fn == True)[0]
                np.add.at(metrics_races[race_comb]['fn_clusters'], cluster_pairs_labels_race_sample0[indices_fn_race], 1)
                np.add.at(metrics_races[race_comb]['fn_clusters'], cluster_pairs_labels_race_sample1[indices_fn_race], 1)

                sum_tp_fp_tn_fn = metrics_races[race_comb]['tp_clusters'] + metrics_races[race_comb]['fp_clusters'] + metrics_races[race_comb]['tn_clusters'] + metrics_races[race_comb]['fn_clusters']
                metrics_races[race_comb]['acc_clusters'] = np.divide(metrics_races[race_comb]['tp_clusters'] + metrics_races[race_comb]['tn_clusters'],
                                                                     sum_tp_fp_tn_fn,
                                                                     where=sum_tp_fp_tn_fn != 0,
                                                                     out=np.zeros_like(metrics_races[race_comb]['tp_clusters']))

                np.add.at(metrics_races[race_comb]['num_samples_clusters'], cluster_pairs_labels_race_sample0, 1)
                np.add.at(metrics_races[race_comb]['num_samples_clusters'], cluster_pairs_labels_race_sample1, 1)

                indices_hits   = np.hstack((indices_tp_race, indices_tn_race))
                indices_misses = np.hstack((indices_fp_race, indices_fn_race))
                for idx_index_hit, index_hit in enumerate(indices_hits):
                    if cluster_pairs_labels_race_sample0[index_hit] == cluster_pairs_labels_race_sample1[index_hit]:
                        metrics_races[race_comb]['num_hits_same_style_clusters'][cluster_pairs_labels_race_sample0[index_hit]] += 1
                        metrics_races[race_comb]['num_hits_same_style_clusters'][cluster_pairs_labels_race_sample1[index_hit]] += 1
                    else:
                        metrics_races[race_comb]['num_hits_diff_style_clusters'][cluster_pairs_labels_race_sample0[index_hit]] += 1
                        metrics_races[race_comb]['num_hits_diff_style_clusters'][cluster_pairs_labels_race_sample1[index_hit]] += 1

                metrics_races[race_comb]['perc_hits_same_style_clusters'] = np.divide(metrics_races[race_comb]['num_hits_same_style_clusters'],
                                                                                      metrics_races[race_comb]['num_samples_clusters'],
                                                                                      where=metrics_races[race_comb]['num_samples_clusters'] != 0,
                                                                                      out=np.zeros_like(metrics_races[race_comb]['num_hits_same_style_clusters']))

                metrics_races[race_comb]['perc_hits_diff_style_clusters'] = np.divide(metrics_races[race_comb]['num_hits_diff_style_clusters'],
                                                                                      metrics_races[race_comb]['num_samples_clusters'],
                                                                                      where=metrics_races[race_comb]['num_samples_clusters'] != 0,
                                                                                      out=np.zeros_like(metrics_races[race_comb]['num_hits_diff_style_clusters']))

    if races_list is None:
        return tpr, fpr, acc, predict_issame
    else:
        return tpr, fpr, acc, metrics_races, predict_issame


def calculate_fnmr_fmr_analyze_races(args, thresholds,
                                    embeddings1,
                                    embeddings2,
                                    actual_issame,
                                    fmr_targets,
                                    races_list,
                                    subj_list,
                                    nrof_folds=10,
                                    races_combs=[]):
    assert (embeddings1.shape[0] == embeddings2.shape[0])
    assert (embeddings1.shape[1] == embeddings2.shape[1])
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = LFold(n_splits=nrof_folds, shuffle=False)

    fnmr = {}
    for fmr_target in fmr_targets:
        fnmr[fmr_target] = np.zeros(nrof_folds)
    fmr = np.zeros(nrof_folds)

    # diff = np.subtract(embeddings1, embeddings2)
    # dist = np.sum(np.square(diff), 1)
    # dist = cosine_dist(embeddings1, embeddings2)
    dist = compute_score(embeddings1, embeddings2, args.score)

    indices = np.arange(nrof_pairs)
    metrics_races = [None] * nrof_folds

    # Bernardo
    dist_fusion = None
    if args.fusion_dist != '' and dist_fusion is None:
        print(f'Loading dist for fusion: \'{args.fusion_dist}\'...')
        dist_fusion = np.load(args.fusion_dist)
        print(f'Fusing scores...\n')
        assert dist.shape[0] == dist_fusion.shape[0]
        dist = fuse_scores(dist, dist_fusion)

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        # Find the threshold that gives FMR = fmr_target
        fmr_train = np.zeros(nrof_thresholds)
        for threshold_idx, threshold in enumerate(thresholds):
            _, fmr_train[threshold_idx] = get_fnmr_fmr_analyze_races(
                args, threshold, dist[train_set], actual_issame[train_set], races_list=None, subj_list=None, races_combs=None)

        f = interpolate.interp1d(fmr_train, thresholds, kind='slinear')
        for fmr_target in fmr_targets:
            threshold = f(fmr_target)
            fnmr[fmr_target][fold_idx], fmr[fold_idx] = get_fnmr_fmr_analyze_races(
                args, threshold, dist[test_set], actual_issame[test_set], races_list=None, subj_list=None, races_combs=None)

    fnmr_mean, fnmr_std = {}, {}
    for fmr_target in fmr_targets:
        fnmr_mean[fmr_target] = np.mean(fnmr[fmr_target])
        fnmr_std[fmr_target] = np.std(fnmr[fmr_target])
    fmr_mean = np.mean(fmr)
    return fnmr_mean, fnmr_std, fmr_mean


def get_fnmr_fmr_analyze_races(args, threshold, dist, actual_issame, races_list, subj_list, races_combs):
    # predict_issame = np.less(dist, threshold)
    predict_issame = get_predict_true(dist, threshold, args.score)

    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))

    fnmr = 0 if (fn + tp == 0) else float(fn) / float(fn + tp)
    fmr = 0  if (fp + tn == 0) else float(fp) / float(fp + tn)

    return fnmr, fmr



def calculate_val_analyze_races(args, thresholds,
                  embeddings1,
                  embeddings2,
                  actual_issame,
                  far_target,
                  races_list,
                  subj_list,
                  nrof_folds=10,
                  races_combs=[]):
    assert (embeddings1.shape[0] == embeddings2.shape[0])
    assert (embeddings1.shape[1] == embeddings2.shape[1])
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = LFold(n_splits=nrof_folds, shuffle=False)

    val = np.zeros(nrof_folds)
    far = np.zeros(nrof_folds)

    # diff = np.subtract(embeddings1, embeddings2)
    # dist = np.sum(np.square(diff), 1)
    # dist = cosine_dist(embeddings1, embeddings2)
    dist = compute_score(embeddings1, embeddings2, args.score)

    indices = np.arange(nrof_pairs)
    metrics_races = [None] * nrof_folds

    # Bernardo
    dist_fusion = None
    if args.fusion_dist != '' and dist_fusion is None:
        print(f'Loading dist for fusion: \'{args.fusion_dist}\'...')
        dist_fusion = np.load(args.fusion_dist)
        print(f'Fusing scores...\n')
        assert dist.shape[0] == dist_fusion.shape[0]
        dist = fuse_scores(dist, dist_fusion)

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        # Find the threshold that gives FAR = far_target
        far_train = np.zeros(nrof_thresholds)
        for threshold_idx, threshold in enumerate(thresholds):
            _, far_train[threshold_idx] = calculate_val_far_analyze_races(
                args, threshold, dist[train_set], actual_issame[train_set], races_list=None, subj_list=None, races_combs=None)
        if np.max(far_train) >= far_target:
            f = interpolate.interp1d(far_train, thresholds, kind='slinear')
            threshold = f(far_target)
        else:
            threshold = 0.0

        if not races_list is None and not subj_list is None:
            val[fold_idx], far[fold_idx], metrics_races[fold_idx] = calculate_val_far_analyze_races(
                args, threshold, dist[test_set], actual_issame[test_set], races_list[test_set], subj_list[test_set], races_combs=races_combs)
        else:
            val[fold_idx], far[fold_idx] = calculate_val_far_analyze_races(
                args, threshold, dist[test_set], actual_issame[test_set], races_list=None, subj_list=None, races_combs=races_combs)

    avg_val_metrics = None
    if not races_list is None and not subj_list is None:
        avg_val_metrics = get_avg_val_metrics_races(metrics_races, races_combs)

    val_mean = np.mean(val)
    far_mean = np.mean(far)
    val_std = np.std(val)
    return val_mean, val_std, far_mean, avg_val_metrics


def calculate_val_far_analyze_races(args, threshold, dist, actual_issame, races_list, subj_list, races_combs):
    # predict_issame = np.less(dist, threshold)
    predict_issame = get_predict_true(dist, threshold, args.score)

    true_accept = np.sum(np.logical_and(predict_issame, actual_issame))
    false_accept = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    n_same = np.sum(actual_issame)
    n_diff = np.sum(np.logical_not(actual_issame))
    # print(true_accept, false_accept)
    # print(n_same, n_diff)
    val = (float(true_accept) / float(n_same))  if n_same > 0 else 0.0
    far = (float(false_accept) / float(n_diff)) if n_diff > 0 else 0.0

    # race analysis (African, Asian, Caucasian, Indian)
    if not races_list is None and not subj_list is None:
        metrics_races = {}
        for race_comb in races_combs:
            metrics_races[race_comb] = {}

        for i, race_comb in enumerate(races_combs):
            indices_race_comb = np.where(np.all(races_list == race_comb, axis=1))[0]
            metrics_races[race_comb]['true_accept'] = np.sum(np.logical_and(predict_issame[indices_race_comb], actual_issame[indices_race_comb]))
            metrics_races[race_comb]['false_accept'] = np.sum(np.logical_and(predict_issame[indices_race_comb], np.logical_not(actual_issame[indices_race_comb])))
            metrics_races[race_comb]['n_same'] = np.sum(actual_issame[indices_race_comb])
            metrics_races[race_comb]['n_diff'] = np.sum(np.logical_not(actual_issame[indices_race_comb]))

            metrics_races[race_comb]['val'] = float(metrics_races[race_comb]['true_accept']) / float(metrics_races[race_comb]['n_same'])
            metrics_races[race_comb]['far'] = float(metrics_races[race_comb]['false_accept']) / float(metrics_races[race_comb]['n_diff'])
    
    if races_list is None:
        return val, far
    else:
        return val, far, metrics_races


def calculate_best_acc(args, thresholds,
                  embeddings1,
                  embeddings2,
                  actual_issame,
                  races_list,
                  subj_list,
                  races_combs=[]):
    assert (embeddings1.shape[0] == embeddings2.shape[0])
    assert (embeddings1.shape[1] == embeddings2.shape[1])
    # nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    # k_fold = LFold(n_splits=nrof_folds, shuffle=False)

    # tprs = np.zeros((nrof_thresholds))
    # fprs = np.zeros((nrof_thresholds))
    accuracy = np.zeros((nrof_thresholds))
    # indices = np.arange(nrof_pairs)
    # metrics_races = [None] * nrof_folds

    # diff = np.subtract(embeddings1, embeddings2)
    # dist = np.sum(np.square(diff), 1)
    # dist = cosine_dist(embeddings1, embeddings2)
    dist = compute_score(embeddings1, embeddings2, args.score)

    # Bernardo
    dist_fusion = None
    if args.fusion_dist != '':
        print(f'Loading dist for fusion: \'{args.fusion_dist}\'...')
        dist_fusion = np.load(args.fusion_dist)
        print(f'Fusing scores...\n')
        assert dist.shape[0] == dist_fusion.shape[0]
        dist = fuse_scores(dist, dist_fusion)

    # Find best threshold
    for threshold_idx, threshold in enumerate(thresholds):
        _, _, accuracy[threshold_idx], _ = calculate_accuracy_analyze_races(
            args, threshold, dist, actual_issame, races_list=None, subj_list=None, races_combs=None, style_clusters_data=None)
    best_threshold_index = np.argmax(accuracy)
    best_threshold = thresholds[best_threshold_index]
    _, _, best_accuracy, _ = calculate_accuracy_analyze_races(
                args, best_threshold, dist, actual_issame, races_list=None, subj_list=None, races_combs=None, style_clusters_data=None)

    return best_accuracy, best_threshold


def calculate_acc_at_threshold(args, one_threshold,
                  embeddings1,
                  embeddings2,
                  actual_issame,
                  races_list,
                  subj_list,
                  races_combs=[]):
    assert (embeddings1.shape[0] == embeddings2.shape[0])
    assert (embeddings1.shape[1] == embeddings2.shape[1])
    # nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    # nrof_thresholds = len(thresholds)
    # k_fold = LFold(n_splits=nrof_folds, shuffle=False)

    # tprs = np.zeros((nrof_thresholds))
    # fprs = np.zeros((nrof_thresholds))
    # accuracy = np.zeros((nrof_thresholds))
    # indices = np.arange(nrof_pairs)
    # metrics_races = [None] * nrof_folds

    # diff = np.subtract(embeddings1, embeddings2)
    # dist = np.sum(np.square(diff), 1)
    # dist = cosine_dist(embeddings1, embeddings2)
    dist = compute_score(embeddings1, embeddings2, args.score)
    predict_issame = get_predict_true(dist, one_threshold, args.score)
    predict_labels_at_thresh = predict_issame.astype(int)

    # Bernardo
    dist_fusion = None
    if args.fusion_dist != '':
        print(f'Loading dist for fusion: \'{args.fusion_dist}\'...')
        dist_fusion = np.load(args.fusion_dist)
        print(f'Fusing scores...\n')
        assert dist.shape[0] == dist_fusion.shape[0]
        dist = fuse_scores(dist, dist_fusion)

    # compute metrics at one_threshold
    _, _, accuracy_at_thresh = calculate_accuracy_analyze_races(
                args, one_threshold, dist, actual_issame, races_list=None, subj_list=None, races_combs=None)

    return accuracy_at_thresh, dist, predict_labels_at_thresh



def save_scores_pred_labels_frcsyn_format(file_path, float_array, int_array):
    if len(float_array) != len(int_array):
        raise ValueError("Both arrays must have the same length")
    with open(file_path, 'w') as file:
        for float_val, int_val in zip(float_array, int_array):
            file.write(f"{float_val},{int_val}\n")


def save_img_pairs(args, actual_issame, predict_issame, dist, idxs_save, imgs, path_folder, chart_title, chart_subtitle='', pair_type=''):
    num_pairs_to_save = args.save_best_worst_pairs
    if args.save_best_worst_pairs < 0:
        num_pairs_to_save = len(idxs_save)

    for idx in range(num_pairs_to_save):
        print(f'{idx}/{num_pairs_to_save}', end='\r')
        if dist[idxs_save[idx]] == -np.inf or dist[idxs_save[idx]] == np.inf:
            break

        image1 = imgs[idxs_save[idx]*2]
        image2 = imgs[idxs_save[idx]*2+1]
        image1 = image1.squeeze(0).permute(1, 2, 0).byte().numpy()
        image2 = image2.squeeze(0).permute(1, 2, 0).byte().numpy()
        
        # Create figure
        plt.clf()
        fig_size=(7, 4)
        fig, axes = plt.subplots(1, 2, figsize=fig_size)
        
        # Display images side by side
        axes[0].imshow(image1)
        axes[0].axis('on')
        axes[1].imshow(image2)
        axes[1].axis('on')
        
        # Add title and subtitle if provided
        fig.suptitle(chart_title + 
                     f'    actual: {str(bool(actual_issame[idxs_save[idx]]))}    pred: {str(bool(predict_issame[idxs_save[idx]]))} ({pair_type})', fontsize=16)
        
        fig.text(0.5, 0.85, chart_subtitle +
                            f'    rank: {str(idx).zfill(7)}    pair-idx: {idxs_save[idx]}    cossim: {dist[idxs_save[idx]]:.3f}',
                            ha='center', fontsize=12)
        
        # Save the figure as a PNG file
        # output_path = os.path.join(path_folder, f'{pair_type}_{str(idx).zfill(5)}_pair={str(idxs_save[idx]).zfill(5)}_cossim={dist[idxs_save[idx]]:.3f}'+'.png')
        # output_path = os.path.join(path_folder, f'{pair_type}_pair={str(idxs_save[idx]).zfill(5)}_{str(idx).zfill(5)}_cossim={dist[idxs_save[idx]]:.3f}'+'.png')
        output_path = os.path.join(path_folder, f'rank={str(idx).zfill(7)}_{pair_type}_pair-idx={str(idxs_save[idx]).zfill(7)}_cossim={dist[idxs_save[idx]]:.3f}'+'.png')
        plt.savefig(output_path, format='png')
        plt.clf()
        plt.close(fig)
    print()



def save_best_and_worst_pairs(args, path_dir_model, thresholds,
                              embeddings1,
                              embeddings2,
                              actual_issame,
                              races_list,
                              subj_list,
                              nrof_folds=10,
                              pca=0,
                              races_combs=[],
                              imgs=[],
                              style_clusters_data={}):
    assert (embeddings1.shape[0] == embeddings2.shape[0])
    assert (embeddings1.shape[1] == embeddings2.shape[1])
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = LFold(n_splits=nrof_folds, shuffle=False)
    predict_issame = np.full_like(actual_issame, 0)     # Bernardo

    tprs = np.zeros((nrof_folds, nrof_thresholds))
    fprs = np.zeros((nrof_folds, nrof_thresholds))
    accuracy = np.zeros((nrof_folds))
    indices = np.arange(nrof_pairs)
    metrics_races = [None] * nrof_folds

    if pca == 0:
        # diff = np.subtract(embeddings1, embeddings2)
        # dist = np.sum(np.square(diff), 1)
        # dist = cosine_dist(embeddings1, embeddings2)
        dist = compute_score(embeddings1, embeddings2, args.score)
        

    # Bernardo
    dist_fusion = None
    if args.fusion_dist != '':
        print(f'Loading dist for fusion: \'{args.fusion_dist}\'...')
        dist_fusion = np.load(args.fusion_dist)
        print(f'Fusing scores...\n')
        assert dist.shape[0] == dist_fusion.shape[0]
        dist = fuse_scores(dist, dist_fusion)

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        if pca > 0:
            print('doing pca on', fold_idx)
            embed1_train = embeddings1[train_set]
            embed2_train = embeddings2[train_set]
            _embed_train = np.concatenate((embed1_train, embed2_train), axis=0)
            pca_model = PCA(n_components=pca)
            pca_model.fit(_embed_train)
            embed1 = pca_model.transform(embeddings1)
            embed2 = pca_model.transform(embeddings2)
            embed1 = sklearn.preprocessing.normalize(embed1)
            embed2 = sklearn.preprocessing.normalize(embed2)
            # diff = np.subtract(embed1, embed2)
            # dist = np.sum(np.square(diff), 1)
            # dist = cosine_dist(embed1, embed2)
            dist = compute_score(embed1, embed1, args.score)

            if not dist_fusion is None:
                print(f'Fusing scores (pca)...')
                assert dist.shape[0] == dist_fusion.shape[0]
                dist = fuse_scores(dist, dist_fusion)

        # Find the best threshold for the fold
        acc_train = np.zeros((nrof_thresholds))
        for threshold_idx, threshold in enumerate(thresholds):
            _, _, acc_train[threshold_idx], _ = calculate_accuracy_analyze_races(
                args, threshold, dist[train_set], actual_issame[train_set], races_list=None, subj_list=None, races_combs=None, style_clusters_data=None)
        best_threshold_index = np.argmax(acc_train)
        for threshold_idx, threshold in enumerate(thresholds):
            tprs[fold_idx, threshold_idx], fprs[fold_idx, threshold_idx], _, predict_issame[test_set] = calculate_accuracy_analyze_races(
                args, threshold, dist[test_set], actual_issame[test_set], races_list=None, subj_list=None, races_combs=None, style_clusters_data=None)
        
        if not races_list is None and not subj_list is None:
            _, _, accuracy[fold_idx], metrics_races[fold_idx], predict_issame[test_set] = calculate_accuracy_analyze_races(
                args, thresholds[best_threshold_index], dist[test_set],
                actual_issame[test_set], races_list[test_set], subj_list[test_set], races_combs=races_combs, style_clusters_data=None)
        
            if not style_clusters_data is None:
                _, _, accuracy[fold_idx], metrics_races[fold_idx], _ = calculate_accuracy_analyze_races(
                    args, thresholds[best_threshold_index], dist[test_set],
                    actual_issame[test_set], races_list[test_set], subj_list[test_set], races_combs=races_combs, style_clusters_data=style_clusters_data)
        
        else:
            _, _, accuracy[fold_idx], predict_issame[test_set] = calculate_accuracy_analyze_races(
                args, thresholds[best_threshold_index], dist[test_set],
                actual_issame[test_set], races_list=None, subj_list=None, races_combs=races_combs, style_clusters_data=None)

    tp = np.logical_and(predict_issame, actual_issame)
    fp = np.logical_and(predict_issame, np.logical_not(actual_issame))
    tn = np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame))
    fn = np.logical_and(np.logical_not(predict_issame), actual_issame)
    # print('tp:', tp, '    np.sum(tp):', np.sum(tp), '    dist[tp]:', dist[tp])   # np.sum(tp): 2905   dist[tp]: [0.40210061 0.79455505 0.68250193 ... 0.56299254 0.80737444 0.57917398]
    # print('fp:', fp, '    np.sum(fp):', np.sum(fp), '    dist[tp]:', dist[fp])   # np.sum(fp): 44     dist[tp]: [0.23618177 0.30582174 0.23749101 0.24565418 0.25256271 0.23503945
    # print('tn:', tn, '    np.sum(tn):', np.sum(tn), '    dist[tp]:', dist[tn])   # np.sum(tn): 2956   dist[tp]: [0.07079436 0.04933054 0.         ... 0.         0.07501368 0.10491953]
    # print('fn:', fn, '    np.sum(fn):', np.sum(fn), '    dist[tp]:', dist[fn])   # np.sum(fn): 95     dist[tp]: [0.22380015 0.19986544 0.18416269 0.05134246 0.         0.1915093
    # sys.exit(0)
    dist_fp = np.full_like(dist, -np.inf); dist_fp[fp] = dist[fp]
    dist_fn = np.full_like(dist,  np.inf); dist_fn[fn] = dist[fn]
    worst_fp_idx = np.argsort(dist_fp)[::-1]
    worst_fn_idx = np.argsort(dist_fn)
    # print('worst_fp:', worst_fp)
    # print(f'dist_fp[worst_fp_idx][:{args.save_best_worst_pairs}]:', dist_fp[worst_fp_idx][:args.save_best_worst_pairs])
    # print('worst_fn:', worst_fn_idx)
    # print(f'dist_fn[worst_fn_idx][:{args.save_best_worst_pairs}]:', dist_fn[worst_fn_idx][:args.save_best_worst_pairs])
    # sys.exit(0)

    path_eval_dataset = os.path.join(path_dir_model, 'eval_pairs_'+args.target)
    if args.protocol: path_eval_dataset += f"_prot={args.protocol.split('/')[-1].split('.')[0]}" 
    print('    path_eval_dataset:', path_eval_dataset)

    pair_type = 'fp'
    path_fp_dataset = os.path.join(path_eval_dataset, pair_type)
    os.makedirs(path_fp_dataset, exist_ok=True)
    chart_title = f'Dataset: {args.target}'
    chart_subtitle = ''
    save_img_pairs(args, actual_issame, predict_issame, dist_fp, worst_fp_idx, imgs, path_fp_dataset, chart_title, chart_subtitle, pair_type)
    
    pair_type = 'fn'
    path_fn_dataset = os.path.join(path_eval_dataset, pair_type)
    os.makedirs(path_fn_dataset, exist_ok=True)
    chart_title = f'Dataset: {args.target}'
    chart_subtitle = ''
    save_img_pairs(args, actual_issame, predict_issame, dist_fn, worst_fn_idx, imgs, path_fn_dataset, chart_title, chart_subtitle, pair_type)
    # sys.exit(0)



def evaluate_analyze_races(args, path_dir_model, embeddings, actual_issame, races_list, subj_list, nrof_folds=10, pca=0, races_combs=[], imgs=[], style_clusters_data={}):
    # Calculate evaluation metrics
    thresholds = np.arange(0, 4, 0.01)
    if args.score == 'cos-sim':
        thresholds = np.flipud(thresholds)
    embeddings1 = embeddings[0::2]
    embeddings2 = embeddings[1::2]
    print('Doing ROC analysis...')
    tpr, fpr, accuracy, avg_roc_metrics = calculate_roc_analyze_races(args, thresholds,
                                                embeddings1,
                                                embeddings2,
                                                np.asarray(actual_issame),
                                                races_list,
                                                subj_list,
                                                nrof_folds=nrof_folds,
                                                pca=pca,
                                                races_combs=races_combs,
                                                style_clusters_data=style_clusters_data)

    thresholds = np.arange(0, 4, 0.001)
    if args.score == 'cos-sim':
        thresholds = np.flipud(thresholds)
    print('Doing TAR@FAR analysis...')
    val, val_std, far, avg_val_metrics = calculate_val_analyze_races(args, thresholds,
                                                embeddings1,
                                                embeddings2,
                                                np.asarray(actual_issame),
                                                1e-3,
                                                races_list,
                                                subj_list,
                                                nrof_folds=nrof_folds,
                                                races_combs=races_combs)

    thresholds = np.arange(0, 4, 0.0001)
    if args.score == 'cos-sim':
        thresholds = np.flipud(thresholds)
    fmr_targets = [1e-2, 1e-3, 1e-4]
    print('Doing FNMR@FMR analysis...')
    fnmr_mean, fnmr_std, fmr_mean = calculate_fnmr_fmr_analyze_races(args, thresholds,
                                                embeddings1,
                                                embeddings2,
                                                np.asarray(actual_issame),
                                                fmr_targets,
                                                races_list,
                                                subj_list,
                                                nrof_folds=nrof_folds,
                                                races_combs=races_combs)
    
    thresholds = np.arange(0, 4, 0.01)
    if args.score == 'cos-sim':
        thresholds = np.flipud(thresholds)
    print('Doing ACC@BEST-THRESH analysis...')
    best_acc, best_thresh = calculate_best_acc(args, thresholds,
                                                embeddings1,
                                                embeddings2,
                                                np.asarray(actual_issame),
                                                races_list,
                                                subj_list,
                                                races_combs=races_combs)

    acc_at_thresh = None
    if args.save_scores_at_thresh > 0:
        one_threshold = args.save_scores_at_thresh
        print('Doing ACC@THRESH analysis...')
        acc_at_thresh, dist, pred_labels_at_thresh = calculate_acc_at_threshold(args, one_threshold,
                                                        embeddings1,
                                                        embeddings2,
                                                        np.asarray(actual_issame),
                                                        races_list,
                                                        subj_list,
                                                        races_combs=races_combs)

        file_scores_labels = args.model.split('/')[-1].split('.')[0] + '_target=' + args.target.split('/')[-1].split('.')[0] + f'_frcsyn_scores_labels_thresh={one_threshold}.txt'
        path_file_scores_labels = os.path.join(os.path.dirname(args.model), file_scores_labels)
        print(f'    Saving scores and pred labels at \'{path_file_scores_labels}\'...')
        save_scores_pred_labels_frcsyn_format(path_file_scores_labels, dist, pred_labels_at_thresh)


    if args.save_best_worst_pairs != 0:
        print('Saving best/worst pairs...')
        save_best_and_worst_pairs(args, path_dir_model, thresholds,
                                  embeddings1,
                                  embeddings2,
                                  np.asarray(actual_issame),
                                  races_list,
                                  subj_list,
                                  nrof_folds=nrof_folds,
                                  pca=pca,
                                  races_combs=races_combs,
                                  imgs=imgs,
                                  style_clusters_data=style_clusters_data)

    print('--------------------')
    return tpr, fpr, accuracy, val, val_std, far, fnmr_mean, fnmr_std, fmr_mean, avg_roc_metrics, avg_val_metrics, \
            best_acc, best_thresh, acc_at_thresh


@torch.no_grad()
def test_analyze_races(args, name, path_dir_model, data_set, backbone, batch_size, nfolds=10, races_combs=[], style_clusters_data={}):
    data_list = data_set[0]
    issame_list = data_set[1]
    # if len(data_set) > 2:
    if name.lower() == 'bupt':
        races_list = data_set[2]
        subj_list = data_set[3]
    else:
        races_list, subj_list = None, None

    os.makedirs(path_dir_model, exist_ok=True)
    # path_embeddings = os.path.join(path_dir_model, 'embeddings_list.pkl')
    path_embeddings = os.path.join(path_dir_model, f"embeddings_list_{args.protocol.split('/')[-1].split('.')[0]}.pkl")

    if not os.path.exists(path_embeddings) or not args.use_saved_embedd:
        print('\nComputing embeddings...')
        embeddings_list = []
        time_consumed = 0.0
        for i in range(len(data_list)):
            data = data_list[i]
            embeddings = None
            ba = 0
            while ba < data.shape[0]:
                bb = min(ba + batch_size, data.shape[0])
                print(f'{i+1}/{len(data_list)} - {bb}/{data.shape[0]}', end='\r')
                count = bb - ba
                _data = data[bb - batch_size: bb]
                time0 = datetime.datetime.now()
                img = ((_data / 255) - 0.5) / 0.5
                # print('img:', img)
                # print('img.size():', img.size())

                net_out: torch.Tensor = backbone(img)             # original
                # net_out: torch.Tensor = backbone.forward(img)   # Bernardo

                _embeddings = net_out.detach().cpu().numpy()
                time_now = datetime.datetime.now()
                diff = time_now - time0
                time_consumed += diff.total_seconds()
                if embeddings is None:
                    embeddings = np.zeros((data.shape[0], _embeddings.shape[1]))
                embeddings[ba:bb, :] = _embeddings[(batch_size - count):, :]
                ba = bb
            embeddings_list.append(embeddings)
            print('')
        print('infer time', time_consumed)
        
        print(f'Saving embeddings in file \'{path_embeddings}\' ...')
        write_object_to_file(path_embeddings, embeddings_list)
    else:
        print(f'Loading embeddings from file \'{path_embeddings}\' ...')
        embeddings_list = read_object_from_file(path_embeddings)

    print(f'Normalizing embeddings...')
    _xnorm = 0.0
    _xnorm_cnt = 0
    for embed in embeddings_list:
        for i in range(embed.shape[0]):
            _em = embed[i]
            _norm = np.linalg.norm(_em)
            _xnorm += _norm
            _xnorm_cnt += 1
    _xnorm /= _xnorm_cnt

    embeddings = embeddings_list[0].copy()
    embeddings = sklearn.preprocessing.normalize(embeddings)
    acc1 = 0.0
    std1 = 0.0
    embeddings = embeddings_list[0] + embeddings_list[1]
    embeddings = sklearn.preprocessing.normalize(embeddings)
    print(embeddings.shape)

    # TESTS
    # print('data_list[0][0]:', data_list[0][0])
    # print('data_list[0].shape:', data_list[0].shape)   # torch.Size([12000, 3, 112, 112])
    # sys.exit(0)
    # idx_img = 0; save_img(f'./image_{idx_img}.png', data_list[0][idx_img])
    # idx_img = 1; save_img(f'./image_{idx_img}.png', data_list[0][idx_img])
    # idx_img = 2; save_img(f'./image_{idx_img}.png', data_list[0][idx_img])
    # idx_img = 3; save_img(f'./image_{idx_img}.png', data_list[0][idx_img])
    # sys.exit(0)

    print('\nDoing races test evaluation...')
    # _, _, accuracy, val, val_std, far = evaluate(embeddings, issame_list, nrof_folds=nfolds)
    _, _, accuracy, val, val_std, far, fnmr_mean, fnmr_std, fmr_mean, avg_roc_metrics, avg_val_metrics, \
        best_acc, best_thresh, acc_at_thresh = evaluate_analyze_races(args, path_dir_model, embeddings, issame_list, races_list, subj_list, nrof_folds=nfolds, races_combs=races_combs, imgs=data_list[0], style_clusters_data=style_clusters_data)
    acc2, std2 = np.mean(accuracy), np.std(accuracy)
    return acc1, std1, acc2, std2, _xnorm, embeddings_list, val, val_std, far, fnmr_mean, fnmr_std, fmr_mean, avg_roc_metrics, avg_val_metrics, \
            best_acc, best_thresh, acc_at_thresh


def read_object_from_file(path):
    with open(path, 'rb') as fid:
        any_obj = pickle.load(fid)
    return any_obj


def write_object_to_file(path, any_obj):
    with open(path, 'wb') as fid:
        # pickle.dump(any_obj, fid)
        pickle.dump(any_obj, fid, protocol=pickle.HIGHEST_PROTOCOL)   # allows file bigger than 4GB



def dumpR(data_set,
          backbone,
          batch_size,
          name='',
          data_extra=None,
          label_shape=None):
    print('dump verification embedding..')
    data_list = data_set[0]
    issame_list = data_set[1]
    embeddings_list = []
    time_consumed = 0.0
    for i in range(len(data_list)):
        data = data_list[i]
        embeddings = None
        ba = 0
        while ba < data.shape[0]:
            bb = min(ba + batch_size, data.shape[0])
            count = bb - ba

            _data = nd.slice_axis(data, axis=0, begin=bb - batch_size, end=bb)
            time0 = datetime.datetime.now()
            if data_extra is None:
                db = mx.io.DataBatch(data=(_data,), label=(_label,))
            else:
                db = mx.io.DataBatch(data=(_data, _data_extra),
                                     label=(_label,))
            model.forward(db, is_train=False)
            net_out = model.get_outputs()
            _embeddings = net_out[0].asnumpy()
            time_now = datetime.datetime.now()
            diff = time_now - time0
            time_consumed += diff.total_seconds()
            if embeddings is None:
                embeddings = np.zeros((data.shape[0], _embeddings.shape[1]))
            embeddings[ba:bb, :] = _embeddings[(batch_size - count):, :]
            ba = bb
        embeddings_list.append(embeddings)
    embeddings = embeddings_list[0] + embeddings_list[1]
    embeddings = sklearn.preprocessing.normalize(embeddings)
    actual_issame = np.asarray(issame_list)
    outname = os.path.join('temp.bin')
    with open(outname, 'wb') as f:
        pickle.dump((embeddings, issame_list),
                    f,
                    protocol=pickle.HIGHEST_PROTOCOL)



def find_index_string_containing_substring(string_list, substring):
    for index, string in enumerate(string_list):
        if substring in string:
            return index
    return -1  # Substring not found in any string


def save_styles_per_race_performance_bars_chart(perf_metrics, global_title, output_path):
    races = list(perf_metrics.keys())
    ndarrays_below = [perf_metrics[race]['perc_hits_diff_style_clusters_mean'] for race in races]
    ndarrays_above = [perf_metrics[race]['perc_hits_same_style_clusters_mean'] for race in races]
    stats = [perf_metrics[race]['acc_clusters_mean_metrics'] for race in races]

    if len(ndarrays_below) != len(races) or len(stats) != len(races):
        raise ValueError("The number of ndarrays_below and stats must match the number of subtitles.")

    global_max = 1.0  # 100%
    n_subplots = len(ndarrays_below)

    subplot_height = 2.5
    subplot_spacing = 1
    fig_height = n_subplots * subplot_height
    
    fig, axes = plt.subplots(n_subplots, 2, figsize=(16, fig_height), constrained_layout=False, 
                              gridspec_kw={"width_ratios": [3, 1]})
    fig.subplots_adjust(hspace=subplot_spacing)
    
    if n_subplots == 1:
        axes = [axes]
    
    fig.suptitle(global_title, fontsize=16, weight='bold')
    
    for i, ((bar_ax, stat_ax), arr_below, arr_above, stat, subtitle) in enumerate(zip(axes, ndarrays_below, ndarrays_above, stats, races)):
        bars_below = bar_ax.bar(range(len(arr_below)), arr_below, color="blue", label='acc_pairs_diff_style')
        bars_above = bar_ax.bar(range(len(arr_above)), arr_above, color="green", label='acc_pairs_same_style', bottom=arr_below)
        
        bar_ax.set_ylim(0, global_max)
        bar_ax.set_yticks([0, global_max])
        bar_ax.set_title(f'{subtitle} (styles)', fontsize=14)
        if i == len(ndarrays_below) - 1:
            bar_ax.set_xlabel("Face Styles", fontsize=12)
        bar_ax.set_ylabel("Accuracy", fontsize=12)

        bar_ax.set_xticks(range(len(arr_below)))
        bar_ax.set_xticklabels(range(len(arr_below)), fontsize=8, rotation=90)
        
        if i == 0:
            # bar_ax.legend(loc="upper left", bbox_to_anchor=(1.05, 1))
            bar_ax.legend(loc="upper right", bbox_to_anchor=(1.0, 1.6))
            # bar_ax.legend(loc="best")

        stat_labels = list(stat.keys())
        stat_values = list(stat.values())
        bars = stat_ax.bar(stat_labels, stat_values, color="orange")
        stat_ax.set_title(f'{subtitle} (statistics)', fontsize=14)
        stat_ax.set_ylim(0, 2)
        stat_ax.set_ylabel("Value", fontsize=10)

        for bar, value in zip(bars, stat_values):
            stat_ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05, f'{value:.3f}', 
                         ha='center', va='bottom', fontsize=14)
    
    plt.savefig(output_path, format='png', bbox_inches='tight')
    plt.close(fig)



def compute_statistical_metrics(performance_values):
    # values_sum = np.sum(performance_values)
    # assert values_sum == 0.0 or (values_sum >= 0.99 and values_sum <= 1.0), f'np.sum(performance_values) is {values_sum}, should be in [0.99, 1.0]'
    stats = {}

    mean, std_dev = np.mean(performance_values), np.std(performance_values)
    # stats['mean'] = mean
    # stats['std'] = std_dev

    num_bins = 10
    bins = np.linspace(performance_values.min(), performance_values.max(), num_bins + 1)
    hist, _ = np.histogram(performance_values, bins=bins)
    probabilities = hist / len(performance_values)
    ent = entropy(probabilities, base=2)
    max_entropy = np.log2(num_bins)
    stats['entropy'] = 1 - (ent / max_entropy)  # normalized_entropy

    # gini_index = 1 - np.sum(performance_values**2)
    # stats['gini'] = gini_index

    cv = 0.0
    if std_dev > 0 and mean > 0:
        cv = std_dev / mean
    stats['cv'] = cv

    uniform_prob = np.ones_like(probabilities) / len(probabilities)
    probabilities = probabilities[probabilities > 0]
    uniform_prob = uniform_prob[:len(probabilities)]
    kl_div = np.sum(probabilities * (np.log2(probabilities) - np.log2(uniform_prob)))
    stats['kl_div'] = kl_div

    return stats


def compute_total_counts(races_styles_clusters_count={}):
    first_race = list(races_styles_clusters_count.keys())[0]
    races_styles_clusters_count_total_races = np.zeros((len(races_styles_clusters_count[first_race]),))
    for idx_race, race in enumerate(list(races_styles_clusters_count.keys())):
        races_styles_clusters_count_total_races += races_styles_clusters_count[race]
    return races_styles_clusters_count_total_races
    # races_styles_clusters_count['total_races'] = races_styles_clusters_count_total_races
    # print('races_styles_clusters_count_total_races.sum():', races_styles_clusters_count_total_races.sum())


def normalize_races_styles_clusters_count(races_styles_clusters_count={}):
    races_styles_clusters_count_normalized = {}
    for idx_race, race in enumerate(list(races_styles_clusters_count.keys())):
        if races_styles_clusters_count[race].sum() > 0.0:
            races_styles_clusters_count_normalized[race] = races_styles_clusters_count[race] / races_styles_clusters_count[race].sum()
        else:
            races_styles_clusters_count_normalized[race] = np.zeros_like(races_styles_clusters_count[race])
    return races_styles_clusters_count_normalized


def save_correlations_per_race_scatter_chart(train_race_count_norm_percs, avg_roc_metrics, global_title, output_path):
    races = list(train_race_count_norm_percs.keys())
    num_races = len(races)
    train_styles_counts = [train_race_count_norm_percs[race] for race in races]
    test_styles_perfs = [avg_roc_metrics[equiv_races[race]]['acc_clusters_mean'] for race in races]
    stats = [avg_roc_metrics[equiv_races[race]]['acc_clusters_mean_corrs'] for race in races]

    fig, axes = plt.subplots(num_races, 2, figsize=(9, 2*num_races), gridspec_kw={'width_ratios': [3, 1]})
    fig.suptitle(global_title)

    # Find global min/max for consistent axis limits
    global_min_x = min(min(arr) for arr in train_styles_counts)
    global_max_x = max(max(arr) for arr in train_styles_counts)
    global_min_y = min(min(arr) for arr in test_styles_perfs)
    global_max_y = max(max(arr) for arr in test_styles_perfs)

    for i, race in enumerate(races):
        ax_scatter = axes[i, 0] if num_races > 1 else axes[0]
        ax_text    = axes[i, 1] if num_races > 1 else axes[1]

        ax_scatter.scatter(train_styles_counts[i], test_styles_perfs[i])
        ax_scatter.set_title(f"{race}")
        ax_scatter.set_xlabel("Train Style Proportion (wrt Uniform Expected Value)", fontsize=9)
        ax_scatter.set_ylabel("Test Style Performances", fontsize=9)
        ax_scatter.grid(True)
        ax_scatter.set_xlim(global_min_x, global_max_x)
        ax_scatter.set_ylim(global_min_y, global_max_y)
        ax_scatter.set_yticks(np.arange(0, 1.1, 0.25))
        
        plt.rcParams["font.family"] = "monospace"
        plt.rcParams["font.monospace"] = ["FreeMono"]
        max_len_key = max([len(key) for key in list(stats[i].keys())])
        stat_text = ''
        for idx_key1, (key1, val1) in enumerate(stats[i].items()):
            if idx_key1 == 0: stat_text += 'TYPE'.ljust(max_len_key+2) + 'CORR'.ljust(7) + 'P-VAL' + '\n'
            stat_text += f'{key1.ljust(max_len_key+1)}'
            for key2, val2 in val1.items():
                rfill = 6 if len(key2) < 0 else 7
                stat_text += ('%.4f' % (val2)).rjust(rfill)
            stat_text += '\n'

        ax_text.set_title(f"{race} - Correlations")
        ax_text.text(0.1, 1.0, stat_text, verticalalignment='top', fontsize=12)
        ax_text.axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_path)
    plt.close(fig)


# BUPT:  ('Asian', 'Asian'), ('Indian', 'Indian'), ('African', 'African'), ('Caucasian', 'Caucasian')
# other: 'asian', 'indian', 'black', 'white', 'middle eastern', 'latino hispanic', 'total_races'
equiv_races = {'asian':('Asian', 'Asian'), 'indian':('Indian', 'Indian'), 'black':('African', 'African'), 'white':('Caucasian', 'Caucasian'), 'total_races':'total_races'}
def evaluate_performance_by_race_face_style(train_races_styles_clusters_count_norm, test_races_styles_clusters_count_norm, avg_roc_metrics, args):
    first_race = list(train_races_styles_clusters_count_norm.keys())[0]
    expected_prob_uniform_dist = 1.0 / len(train_races_styles_clusters_count_norm[first_race])
    train_race_count_norm_percs = {}

    for idx_race, race in enumerate(list(test_races_styles_clusters_count_norm.keys())):
        train_race_count_norm = train_races_styles_clusters_count_norm[race]
        test_race_count_norm  = test_races_styles_clusters_count_norm[race]
        acc_clusters_mean_corrs = {}

        if race in list(equiv_races.keys()):
            equiv_race = equiv_races[race]

            train_race_count_norm_perc = train_race_count_norm / expected_prob_uniform_dist
            train_race_count_norm_perc_clip = np.clip(train_race_count_norm_perc, 0.0, 1.0)
            train_race_count_norm_percs[race] = train_race_count_norm_perc
            # train_race_count_norm_percs[race] = train_race_count_norm_perc_clip

            acc_clusters_mean = avg_roc_metrics[equiv_race]['acc_clusters_mean']

            stat_normal_acc_clusters_mean, pvalue_normal_acc_clusters_mean                     = stats.normaltest(acc_clusters_mean)
            stat_normal_train_race_count_norm_percs, pvalue_normal_train_race_count_norm_percs = stats.normaltest(train_race_count_norm_percs[race])
            correlation_pearson, pvalue_pearson   = stats.pearsonr(train_race_count_norm_percs[race], acc_clusters_mean)
            correlation_spearman, pvalue_spearman = stats.spearmanr(train_race_count_norm_percs[race], acc_clusters_mean)
            correlation_kendall, pvalue_kendall   = stats.kendalltau(train_race_count_norm_percs[race], acc_clusters_mean)

            acc_clusters_mean_corrs['train~N']  = {'corr':stat_normal_acc_clusters_mean,  'pval':pvalue_normal_acc_clusters_mean}
            acc_clusters_mean_corrs['test~N']   = {'corr':stat_normal_train_race_count_norm_percs,  'pval':pvalue_normal_train_race_count_norm_percs}
            acc_clusters_mean_corrs['pearson']  = {'corr':correlation_pearson,  'pval':pvalue_pearson}
            acc_clusters_mean_corrs['spearman'] = {'corr':correlation_spearman, 'pval':pvalue_spearman}
            acc_clusters_mean_corrs['kendall']  = {'corr':correlation_kendall,  'pval':pvalue_kendall}
            
            msg_str = f'{race}'
            for key1, val1 in acc_clusters_mean_corrs.items():
                msg_str += f' - {key1}'
                for key2, val2 in val1.items():
                    msg_str += f' {key2}: {val2}'
            print(msg_str)

            # test_race_performance = np.zeros_like(test_race_count_norm)
            # for idx_cluster in range(len(test_race_performance)):
            #     test_race_performance = train_race_count_norm[idx_cluster] / expected_prob_uniform_dist

            avg_roc_metrics[equiv_race]['acc_clusters_mean_corrs'] = acc_clusters_mean_corrs

    global_title = f"Correlations Between Face Styles Count and Style Performances\nDataset={args.target} - nclusters={len(next(iter(test_races_styles_clusters_count_norm.values())))}"
    output_dir = os.path.dirname(args.model)
    output_path = os.path.join(output_dir, f"correlations_face_styles_count_and_performances_by_race_dataset={args.target}_nclusters={len(next(iter(test_races_styles_clusters_count_norm.values())))}.png")
    print(f"Saving correlations chart: '{output_path}'")
    save_correlations_per_race_scatter_chart(train_race_count_norm_percs, avg_roc_metrics, global_title, output_path)




if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='do verification')
    # general
    # parser.add_argument('--data-dir', default='', help='')                                                                                   # original
    parser.add_argument('--data-dir', default='/datasets1/bjgbiesseck/MS-Celeb-1M/faces_emore', help='')                                     # Bernardo
    # parser.add_argument('--data-dir', default='/datasets2/1st_frcsyn_wacv2024/datasets/real/3_BUPT-BalancedFace/race_per_7000_crops_112x112', help='')   # Bernardo
    parser.add_argument('--data-dir2', default='', help='For protocols that uses more than one dataset')   # /datasets1/bjgbiesseck/MICA/FRGC/images_DETECTED_FACES_RETINAFACE_scales=[0.5]_nms=0.4/imgs

    # parser.add_argument('--network', default='r100', type=str, help='')   # default
    parser.add_argument('--network', default='r50', type=str, help='')      # Bernardo
    parser.add_argument('--model',
                        # default='../trained_models/ms1mv3_arcface_r100_fp16/backbone.pth',          # Bernardo
                        default='../work_dirs/casia_frcsyn_r100/2023-10-14_09-51-11_GPU0/model.pt',   # (Trained only on CASIA-Webface)   Bernardo
                        help='path to load model.')
    parser.add_argument('--target',
                        # default='lfw,cfp_ff,cfp_fp,agedb_30',          # original
                        # default='lfw,cfp_fp,agedb_30',                 # original
                        # default='lfw',                                 # Bernardo
                        default='bupt',                                  # Bernardo (hda_doppelganger, doppelver_doppelganger, doppelver_vise, 3d_tec, nd_twins)
                        help='test targets.')
    parser.add_argument('--protocol', default='', type=str, help='')     # /datasets2/1st_frcsyn_wacv2024/comparison_files/comparison_files/sub-tasks_1.1_1.2/bupt_comparison.txt
    parser.add_argument('--gpu', default=0, type=int, help='gpu id')
    parser.add_argument('--batch-size', default=32, type=int, help='')
    parser.add_argument('--max', default='', type=str, help='')
    parser.add_argument('--mode', default=0, type=int, help='')
    parser.add_argument('--nfolds', default=10, type=int, help='')
    parser.add_argument('--use-saved-embedd', action='store_true')

    parser.add_argument('--fusion-dist', type=str, default='', help='')                 # Bernardo
    parser.add_argument('--score', default='cos-sim', type=str, help='')                # Bernardo ('cos-sim', 'cos-dist' or 'eucl-dist')
    parser.add_argument('--save-scores-at-thresh', type=float, default=-1.0, help='')   # Bernardo (0.5)

    parser.add_argument('--save-best-worst-pairs', default=0, type=int)

    parser.add_argument('--test-style-clusters-data', default='', type=str)    # /datasets2/1st_frcsyn_wacv2024/datasets/real/3_BUPT-BalancedFace/race_per_7000_crops_112x112_JUST-PROTOCOL-IMGS_STYLE_FEATURES_CLUSTERING_FROM_1_CASIA-WebFace-imgs_crops_112x112_STYLE_FEATURES_CLUSTERING-feature=_style-_distance=cosine-nclusters=100/feature=_style/_distance=cosine/nclusters=100/clusters-data_feature=_style.pt_distance=cosine_nclusters=100.pkl
    parser.add_argument('--train-style-clusters-data', default='', type=str)   # /datasets2/1st_frcsyn_wacv2024/datasets/real/1_CASIA-WebFace/imgs_crops_112x112_STYLE_FEATURES_CLUSTERING/feature=_style/_distance=cosine/nclusters=100/clusters-data_feature=_style.pt_distance=cosine_nclusters=100.pkl

    args = parser.parse_args()


    image_size = [112, 112]
    print('image_size', image_size)

    ctx = mx.gpu(args.gpu)   # original
    # ctx = mx.cpu()         # Bernardo

    nets = []
    vec = args.model.split(',')
    prefix = args.model.split(',')[0]
    epochs = []

    # LOADING MODEL WITH PYTORCH
    nets = []
    time0 = datetime.datetime.now()
    print(f'Loading trained model \'{args.model}\'...')
    weight = torch.load(args.model)
    resnet = get_model(args.network, dropout=0, fp16=False).cuda()
    resnet.load_state_dict(weight)
    model = torch.nn.DataParallel(resnet)
    model.eval()
    nets.append(model)
    time_now = datetime.datetime.now()
    diff = time_now - time0
    print('model loading time', diff.total_seconds())


    ver_list = []
    ver_name_list = []
    for name in args.target.split(','):

        # Bernardo
        time0 = datetime.datetime.now()
        print('\ndataset name:', name)
        print('args.data_dir:', args.data_dir)

        path = os.path.join(args.data_dir, name + ".bin")
        if os.path.exists(path):
            print('loading.. ', name)
            data_set = load_bin(path, image_size)
            ver_list.append(data_set)
            ver_name_list.append(name)
            # sys.exit(0)
        
        else:
            if name.lower() == 'bupt':
                path_unified_dataset = os.path.join(args.data_dir, 'dataset.pkl')
                if not os.path.exists(path_unified_dataset):
                    print(f'Loading individual images from folder \'{args.data_dir}\' ...')
                    data_set = Loader_BUPT().load_dataset(args.protocol, args.data_dir, image_size)
                    print(f'Saving dataset in file \'{path_unified_dataset}\' ...')
                    write_object_to_file(path_unified_dataset, data_set)
                else:
                    print(f'Loading dataset from file \'{path_unified_dataset}\' ...')
                    data_set = read_object_from_file(path_unified_dataset)

            elif name.lower() == 'hda_doppelganger':
                # raise Exception(f'Evaluation for dataset \'{name.lower()}\' is under construction')
                path_unified_dataset = os.path.join(args.data_dir, f'dataset_{name.lower()}.pkl')
                if not os.path.exists(path_unified_dataset):
                    print(f'Loading individual images from folder \'{args.data_dir}\' ...')
                    data_set = Loader_HDA_Doppelganger().load_dataset(args.data_dir, args.data_dir2, image_size)
                    print(f'Saving dataset in file \'{path_unified_dataset}\' ...')
                    write_object_to_file(path_unified_dataset, data_set)
                else:
                    print(f'Loading dataset from file \'{path_unified_dataset}\' ...')
                    data_set = read_object_from_file(path_unified_dataset)

            elif name.lower() == 'doppelver_doppelganger' or name.lower() == 'doppelver_vise':
                # raise Exception(f'Evaluation for dataset \'{name.lower()}\' is under construction')
                path_unified_dataset = os.path.join(args.data_dir, f'dataset_{name.lower()}.pkl')
                if not os.path.exists(path_unified_dataset):
                    print(f'Loading individual images from folder \'{args.data_dir}\' ...')
                    data_set = Loader_DoppelVer().load_dataset(args.protocol, args.data_dir, image_size)
                    data_set[0][1] = None   # remove flipped images due its huge size (41GB)
                    print(f'Saving dataset in file \'{path_unified_dataset}\' ...')
                    write_object_to_file(path_unified_dataset, data_set)
                else:
                    print(f'Loading dataset from file \'{path_unified_dataset}\' ...')
                    data_set = read_object_from_file(path_unified_dataset)

                print('Flipping images...')
                if type(data_set[0]) is list and data_set[0][1] == None:
                    data_set[0][1] = torch.flip(data_set[0][0], dims=[3])

            elif name.lower() == '3d_tec':
                # raise Exception(f'Evaluation for dataset \'{name.lower()}\' is under construction')
                protocol_file_name = args.protocol.split('/')[-1].split('.')[0]
                path_unified_dataset = os.path.join(args.data_dir, f'dataset_{name.lower()}_{protocol_file_name}.pkl')
                if not os.path.exists(path_unified_dataset):
                    print(f'Loading individual images from folder \'{args.data_dir}\' ...')
                    data_set = Loader_3DTEC().load_dataset(args.protocol, args.data_dir, image_size, only_twins=True)
                    data_set[0][1] = None   # remove flipped images due its possible huge size
                    print(f'Saving dataset in file \'{path_unified_dataset}\' ...')
                    write_object_to_file(path_unified_dataset, data_set)
                else:
                    print(f'Loading dataset from file \'{path_unified_dataset}\' ...')
                    data_set = read_object_from_file(path_unified_dataset)

                print('Flipping images...')
                if type(data_set[0]) is list and data_set[0][1] == None:
                    data_set[0][1] = torch.flip(data_set[0][0], dims=[3])

            elif name.lower() == 'nd_twins':
                # raise Exception(f'Evaluation for dataset \'{name.lower()}\' is under construction')
                protocol_file_name = args.protocol.split('/')[-1].split('.')[0]
                path_unified_dataset = os.path.join(args.data_dir, f'dataset_{name.lower()}_{protocol_file_name}.pkl')
                if not os.path.exists(path_unified_dataset):
                    print(f'Loading individual images from folder \'{args.data_dir}\' ...')
                    data_set = Loader_NDTwins().load_dataset(args.protocol, args.data_dir, image_size, only_twins=True)
                    data_set[0][1] = None   # remove flipped images due its possible huge size
                    print(f'Saving dataset in file \'{path_unified_dataset}\' ...')
                    write_object_to_file(path_unified_dataset, data_set)
                else:
                    print(f'Loading dataset from file \'{path_unified_dataset}\' ...')
                    data_set = read_object_from_file(path_unified_dataset)

                print('Flipping images...')
                if type(data_set[0]) is list and data_set[0][1] == None:
                    data_set[0][1] = torch.flip(data_set[0][0], dims=[3])

            else:
                raise Exception(f'Error, no \'.bin\' file found in \'{args.data_dir}\'')

            ver_list.append(data_set)
            ver_name_list.append(name)
            # print('data_set:', data_set)
            # sys.exit(0)

        test_style_clusters_data = None
        if args.test_style_clusters_data:
            assert len(data_set) > 4, f"len(data_set) == {len(data_set)}, it doesn't have the paths of protocol images. Delete or rename the file '{path_unified_dataset}' and re-run this code."
            samples_orig_paths_list = data_set[4]

            print(f'Loading test-subj-clusters: \'{args.test_style_clusters_data}\'')
            test_style_clusters_data = load_dict(args.test_style_clusters_data)
            print('Loaded test_style_clusters_data.keys():', test_style_clusters_data.keys())

            # dict_keys(['files_paths', 'original_feats', 'cluster_ids', 'feats_tsne', 'cluster_centers_tsne', 'facial_attribs_paths', 'facial_attribs', 'dominant_races', 'races_styles_clusters_count', 'corresp_imgs_paths'])
            # print("test_style_clusters_data['corresp_imgs_paths']:", test_style_clusters_data['corresp_imgs_paths'])
            # print("len(test_style_clusters_data['corresp_imgs_paths']):", len(test_style_clusters_data['corresp_imgs_paths']))
            # print("test_style_clusters_data['corresp_imgs_paths'][0]:", test_style_clusters_data['corresp_imgs_paths'][0])

            test_style_clusters_pairs_labels = []
            test_style_clusters_data_corresp_imgs_paths = test_style_clusters_data['corresp_imgs_paths']
            style_clusters_ids = test_style_clusters_data['cluster_ids']
            num_clusters = len(test_style_clusters_data['cluster_centers_tsne'])
            for idx_pair_orig_paths, pair_orig_paths in enumerate(samples_orig_paths_list):
                print(f"Loading samples clusters labels {idx_pair_orig_paths}/{len(samples_orig_paths_list)}", end='\r')
                # print('pair_orig_paths:', pair_orig_paths)
                sample0, _ = os.path.splitext(pair_orig_paths[0])
                sample1, _ = os.path.splitext(pair_orig_paths[1])

                sample0_idx_cluster_list = find_index_string_containing_substring(test_style_clusters_data_corresp_imgs_paths, sample0)
                assert sample0_idx_cluster_list > -1, f"Error, substring of sample0 not found: '{sample0}'"
                # print('sample0_idx_cluster_list:', sample0_idx_cluster_list)
                # print(f"test_style_clusters_data_corresp_imgs_paths[{sample0_idx_cluster_list}]:", test_style_clusters_data_corresp_imgs_paths[sample0_idx_cluster_list])
                sample1_idx_cluster_list = find_index_string_containing_substring(test_style_clusters_data_corresp_imgs_paths, sample1)
                assert sample1_idx_cluster_list > -1, f"Error, substring of sample1 not found: '{sample1}'"
                
                sample0_cluster_label = int(style_clusters_ids[sample0_idx_cluster_list])
                sample1_cluster_label = int(style_clusters_ids[sample1_idx_cluster_list])
                assert sample0_cluster_label < num_clusters, f"Error, cluster label of sample0 ({sample0_cluster_label}) > num_clusters ({num_clusters})"
                assert sample1_cluster_label < num_clusters, f"Error, cluster label of sample1 ({sample1_cluster_label}) > num_clusters ({num_clusters})"
                style_clusters_pair_labels = (sample0_cluster_label, sample1_cluster_label)
                # print('style_clusters_pair_labels:', style_clusters_pair_labels)
                test_style_clusters_pairs_labels.append(style_clusters_pair_labels)
            print()
            assert len(test_style_clusters_pairs_labels) == len(samples_orig_paths_list)
            # test_style_clusters_data['pairs_cluster_ids'] = test_style_clusters_pairs_labels
            test_style_clusters_data['pairs_cluster_ids'] = np.array(test_style_clusters_pairs_labels)
    
        time_now = datetime.datetime.now()
        diff = time_now - time0
        print('dataset loading time: %.2fs, %.2fm, %.2fh' % (diff.total_seconds(), diff.total_seconds()/60, diff.total_seconds()/3600))
    # sys.exit(0)

    if args.mode == 0:
        for i in range(len(ver_list)):
            results = []
            for model in nets:

                if name.lower() == 'bupt':
                    races_combs = get_races_combinations()
                else:
                    races_combs = None
                
                path_dir_model = os.path.join(os.path.dirname(args.model), f'eval_{name.lower()}')

                acc1, std1, acc2, std2, xnorm, embeddings_list, val, val_std, far, fnmr_mean, fnmr_std, fmr_mean, avg_roc_metrics, avg_val_metrics, \
                        best_acc, best_thresh, acc_at_thresh = test_analyze_races(args, name.lower(), path_dir_model, ver_list[i], model, args.batch_size, args.nfolds, races_combs, test_style_clusters_data)
                results.append(acc2)
                print('[%s]XNorm: %f' % (ver_name_list[i], xnorm))
                # print('[%s]Accuracy: %1.5f+-%1.5f' % (ver_name_list[i], acc1, std1))
                print('[%s]Accuracy-Flip: %1.5f+-%1.5f' % (ver_name_list[i], acc2, std2))
                print('[%s]TAR: %1.5f+-%1.5f    FAR: %1.5f' % (ver_name_list[i], val, val_std, far))

                for fmr_target in list(fnmr_mean.keys()):
                    print('[%s]FNMR: %1.5f+-%1.5f   FMR: %1.5f' % (ver_name_list[i], fnmr_mean[fmr_target], fnmr_std[fmr_target], fmr_target))

                if not races_combs is None:
                    for race_comb in races_combs:
                        race_comb_str = str((race_comb[0][:5], race_comb[1][:5]))
                        print('[%s]Acc %s: %1.5f+-%1.5f' % (ver_name_list[i], race_comb_str, avg_roc_metrics[race_comb]['acc_mean'], avg_roc_metrics[race_comb]['acc_std']), end='    ')
                        print('[%s]TAR %s: %1.5f+-%1.5f' % (ver_name_list[i], race_comb_str, avg_val_metrics[race_comb]['val_mean'], avg_val_metrics[race_comb]['val_std']), end='    ')
                        print('[%s]FAR %s: %1.5f+-%1.5f' % (ver_name_list[i], race_comb_str, avg_val_metrics[race_comb]['far_mean'], avg_val_metrics[race_comb]['far_std']))

                print('[%s]Best Acc: %1.5f    @best_thresh: %1.5f' % (ver_name_list[i], best_acc, best_thresh))
                if not acc_at_thresh is None:
                    print('[%s]Accuracy: %1.5f    @thresh: %1.5f' % (ver_name_list[i], acc_at_thresh, args.save_scores_at_thresh))


                if not test_style_clusters_data is None:
                    print('Computing distributions statiscs...')
                    for idx_race, race in enumerate(list(avg_roc_metrics.keys())):
                        avg_roc_metrics[race]['acc_clusters_mean_metrics'] = compute_statistical_metrics(avg_roc_metrics[race]['acc_clusters_mean'])
                        # print(f"{race}: {avg_roc_metrics[race]}")

                    global_title = f"Face Verification by Race and Face Style Cluster - Dataset={args.target}"
                    output_dir = os.path.dirname(args.model)
                    output_path = os.path.join(output_dir, f"accuracies_by_race_and_face_style_cluster_dataset={args.target}_nclusters={len(test_style_clusters_data['cluster_centers_tsne'])}.png")
                    print(f"Saving accuracies chart: '{output_path}'")
                    save_styles_per_race_performance_bars_chart(avg_roc_metrics, global_title, output_path)

                    
                    train_style_clusters_data = None
                    if args.train_style_clusters_data:
                        # assert len(data_set) > 4, f"len(data_set) == {len(data_set)}, it doesn't have the paths of protocol images. Delete or rename the file '{path_unified_dataset}' and re-run this code."
                        # samples_orig_paths_list = data_set[4]

                        print(f'Loading train-subj-clusters: \'{args.train_style_clusters_data}\'')
                        train_style_clusters_data = load_dict(args.train_style_clusters_data)
                        print('Loaded train_style_clusters_data.keys():', train_style_clusters_data.keys())

                        train_races_styles_clusters_count = train_style_clusters_data['races_styles_clusters_count']
                        # print('train_races_styles_clusters_count:', train_races_styles_clusters_count)
                        if not 'total_races' in list(train_races_styles_clusters_count.keys()):
                            print(f'\nCounting train total face styles...')
                            train_races_styles_clusters_count['total_races'] = compute_total_counts(train_races_styles_clusters_count)
                        print(f'Normalizing train total face styles...')
                        train_races_styles_clusters_count_norm = normalize_races_styles_clusters_count(train_races_styles_clusters_count)

                        test_races_styles_clusters_count = test_style_clusters_data['races_styles_clusters_count']
                        # print('test_races_styles_clusters_count:', test_races_styles_clusters_count)
                        if not 'total_races' in list(test_races_styles_clusters_count.keys()):
                            print(f'\nCounting test total face styles...')
                            test_races_styles_clusters_count['total_races'] = compute_total_counts(test_races_styles_clusters_count)
                        print(f'Normalizing test total face styles...')
                        test_races_styles_clusters_count_norm = normalize_races_styles_clusters_count(test_races_styles_clusters_count)

                        test_performance_by_race_face_style = evaluate_performance_by_race_face_style(train_races_styles_clusters_count_norm,
                                                                                                      test_races_styles_clusters_count_norm,
                                                                                                      avg_roc_metrics,
                                                                                                      args)



            # print('Max of [%s] is %1.5f' % (ver_name_list[i], np.max(results)))
    elif args.mode == 1:
        raise ValueError
    else:
        model = nets[0]
        dumpR(ver_list[0], model, args.batch_size, args.target)
