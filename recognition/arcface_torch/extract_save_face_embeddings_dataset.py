import os, sys
import argparse

import cv2
import numpy as np
import torch
import re
import time

from backbones import get_model


def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--network', type=str, default='r100', help='backbone network')
    parser.add_argument('--weight', type=str, default='./trained_models/ms1mv3_arcface_r100_fp16/backbone.pth')
    parser.add_argument('--imgs', type=str, default='/datasets2/1st_frcsyn_wacv2024/datasets/real/1_CASIA-WebFace/imgs_crops_112x112')
    parser.add_argument('--output-path', type=str, default='')
    parser.add_argument('--start-idx', type=int, default=0)
    args = parser.parse_args()
    return args


def load_trained_model(network, path_weights, device="cuda"):
    net = get_model(network, fp16=False)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = net.to(device)
    print('device:', next(net.parameters()).device)
    # raise Exception()
    net.load_state_dict(torch.load(path_weights))
    net.eval()
    return net


def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


def get_all_files_in_path(folder_path, file_extension=['.jpg','.jpeg','.png'], pattern=''):
    file_list = []
    for root, _, files in os.walk(folder_path):
        for filename in files:
            path_file = os.path.join(root, filename)
            for ext in file_extension:
                if pattern in path_file and path_file.lower().endswith(ext.lower()):
                    file_list.append(path_file)
                    print(f'Found files: {len(file_list)}', end='\r')
    print()
    file_list = natural_sort(file_list)
    return file_list


def load_normalize_img(img, device="cuda"):
    img = cv2.imread(img)
    img = cv2.resize(img, (112, 112))

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.transpose(img, (2, 0, 1))
    img = torch.from_numpy(img).unsqueeze(0).float()
    img = img.to(device)
    img.div_(255).sub_(0.5).div_(0.5)
    return img


@torch.no_grad()
def get_face_embedd(model, img):
    # embedd = model(img).numpy()
    embedd = model(img).cpu().numpy()
    return embedd



if __name__ == '__main__':
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f'Loading trained model ({args.network}): {args.weight}')
    model = load_trained_model(args.network, args.weight, device)
    print()

    args.imgs = args.imgs.rstrip('/')
    if not args.output_path:
        args.output_path = args.imgs + '_FACE_EMBEDDINGS'
    os.makedirs(args.output_path, exist_ok=True)

    print(f'Searching images in \'{args.imgs}\'')
    imgs_paths = get_all_files_in_path(args.imgs)
    print(f'Found {len(imgs_paths)} images\n------------------\n')

    total_elapsed_time = 0.0
    for idx_path, path_img in enumerate(imgs_paths):
        if idx_path >= args.start_idx:
            start_time = time.time()
            print(f'{idx_path}/{len(imgs_paths)} - Computing face embedding')
            print(f'path_img: {path_img}')
            img = load_normalize_img(path_img, device)

            id_feat_img = get_face_embedd(model, img)

            output_path_dir = os.path.dirname(path_img.replace(args.imgs, args.output_path))
            print(f'output_path_dir: {output_path_dir}')
            os.makedirs(output_path_dir, exist_ok=True)

            img_name, img_ext = os.path.splitext(os.path.basename(path_img))
            # output_path_id_feat = os.path.join(output_path_dir, img_name+'_id_feat.pt')
            output_path_id_feat = os.path.join(output_path_dir, img_name+'_embedding_r100_arcface.npy')

            print('output_path_id_feat:', output_path_id_feat)
            if output_path_id_feat.endswith('.pt'):
                torch.save(id_feat_img, output_path_id_feat)
            elif output_path_id_feat.endswith('.npy'):
                np.save(output_path_id_feat, id_feat_img)                

            elapsed_time = time.time()-start_time
            total_elapsed_time += elapsed_time
            avg_sample_time = total_elapsed_time / ((idx_path-args.start_idx)+1)
            estimated_time = avg_sample_time * (len(imgs_paths)-(idx_path+1))
            print("Elapsed time: %.3fs" % elapsed_time)
            print("Avg elapsed time: %.3fs" % avg_sample_time)
            print("Total elapsed time: %.3fs,  %.3fm,  %.3fh" % (total_elapsed_time, total_elapsed_time/60, total_elapsed_time/3600))
            print("Estimated Time to Completion (ETC): %.3fs,  %.3fm,  %.3fh" % (estimated_time, estimated_time/60, estimated_time/3600))
            print('--------------')

        else:
            print(f'Skipping indices: {idx_path}/{len(imgs_paths)}', end='\r')

    print('\nFinished!')
