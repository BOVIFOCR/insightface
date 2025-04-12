import os, sys
import argparse

import cv2
import numpy as np
import torch
import re
import time
import glob

import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--embeddings', type=str, default='/datasets1/bjgbiesseck/doppelgangers_lookalikes/DoppelVer/Images/CCA_Images_DETECTED_FACES_RETINAFACE_scales=[1.0,0.5,0.25]_nms=0.4/imgs_FACE_EMBEDDINGS')
    parser.add_argument('--embedd-ext', type=str, default='_id_feat.pt')
    parser.add_argument('--corresp-imgs', type=str, default='/datasets1/bjgbiesseck/doppelgangers_lookalikes/DoppelVer/Images/CCA_Images_DETECTED_FACES_RETINAFACE_scales=[1.0,0.5,0.25]_nms=0.4/imgs')
    parser.add_argument('--pattern', type=str, default='')
    parser.add_argument('--output-path', type=str, default='')
    args = parser.parse_args()
    return args


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
                    # print(f'Found files: {len(file_list)}', end='\r')
    # print()
    file_list = natural_sort(file_list)
    return file_list


def get_all_files_in_path_by_subj(folder_path, file_extension=['.jpg','.jpeg','.png'], pattern=''):
    files_by_subj = {}
    total_found_files = 0

    for root, dirs, files in os.walk(folder_path):
        if root == folder_path:
            continue
        
        subj = os.path.basename(root)
        matched_files = get_all_files_in_path(root, file_extension)
        total_found_files += len(matched_files)
        if matched_files:
            files_by_subj[subj] = matched_files
        print(f'Found subj: {len(files_by_subj)}    embedds: {total_found_files}', end='\r')
    print()
    return files_by_subj


def get_all_corresponding_imgs_paths_by_subj(corresp_imgs_path='', img_extension=['.jpg','.jpeg','.png'], dict_subj_embedds_paths={}, embeddings_path='', embedd_ext='_id_feat.pt'):
    corresp_imgs_by_subj = {}
    total_found_files = 0
    for idx_subj, (subj_name, subj_embedds_paths) in enumerate(dict_subj_embedds_paths.items()):
        subj_found_corresp_imgs_paths = []
        for idx_embedd, embedd_path in enumerate(subj_embedds_paths):
            # print('idx_subj:', idx_subj, '    subj_name:', subj_name, '    embedd_path:', embedd_path)
            img_pattern = glob.escape(embedd_path.split(args.embedd_ext)[0].replace(embeddings_path,corresp_imgs_path) ) + '*'
            # print('    img_pattern:', img_pattern)
            found_corresp_img_path = glob.glob(img_pattern)
            # print('    found_corresp_img_path:', found_corresp_img_path)
            assert len(found_corresp_img_path) > 0, f'Error, no corresponding image found with patter \'{img_pattern}\' for embedding \'{embedd_path}\''
            assert len(found_corresp_img_path) < 2, f'Error, more than 1 image found with the \'{img_pattern}\' for embedding \'{embedd_path}\''
            subj_found_corresp_imgs_paths.extend(found_corresp_img_path)
            total_found_files += len(found_corresp_img_path)
        corresp_imgs_by_subj[subj_name] = subj_found_corresp_imgs_paths
        print(f'Found corresponding imgs: {total_found_files}', end='\r')
    print()
    return corresp_imgs_by_subj
    # sys.exit(0)


def cosine_similarity(embedd1, embedd2):
    # if len(embedd1.shape) == 1: embedd1 = np.expand_dims(embedd1, axis=0)
    # if len(embedd2.shape) == 1: embedd2 = np.expand_dims(embedd2, axis=0)
    embedd1 = np.expand_dims(embedd1.squeeze(), axis=0)
    embedd2 = np.expand_dims(embedd2.squeeze(), axis=0)
    embedd1 /= np.linalg.norm(embedd1)
    embedd2 /= np.linalg.norm(embedd2)
    sim = float(np.maximum(np.dot(embedd1,embedd2.T)/(np.linalg.norm(embedd1)*np.linalg.norm(embedd2)), 0.0))
    return sim


def save_figure_with_inliers_outliers_faces(
    corresp_imgs_paths,
    corresp_imgs,
    similarities_to_avg_embedd,
    indexes_inliers,
    indexes_outliers,
    title_inliers_outliers,
    path_inliers_outliers
):
    """
    Save a figure with 2 rows: inliers (top) and outliers (bottom), with image titles and row labels.
    """
    n_cols = max(len(indexes_inliers), len(indexes_outliers), 1)
    img_w = 2
    img_h = 2.5

    fig, axes = plt.subplots(
        2, n_cols,
        figsize=(n_cols * img_w, 2 * img_h),
        gridspec_kw={'hspace': 0.8, 'wspace': 0.05}
    )
    fig.suptitle(title_inliers_outliers, fontsize=14)

    def plot_images(indexes, row_idx, label):
        for col_idx in range(n_cols):
            ax = axes[row_idx, col_idx] if n_cols > 1 else axes[row_idx]
            ax.axis('off')

            if col_idx < len(indexes):
                idx = indexes[col_idx]
                img = corresp_imgs[idx]
                filename = os.path.basename(corresp_imgs_paths[idx])
                sim = similarities_to_avg_embedd[idx]
                ax.imshow(img)
                ax.set_title(f"{filename}\n{sim:.2f}", fontsize=5)

        # Add row label on the far left
        row_ax = axes[row_idx, 0] if n_cols > 1 else axes[row_idx]
        row_ax.annotate(
            label,
            xy=(-0.3, 0.5),
            xycoords='axes fraction',
            fontsize=12,
            ha='right',
            va='center',
            rotation=90,
            weight='bold'
        )

    # Make sure axes is 2D
    if n_cols == 1:
        axes = np.array(axes).reshape(2, 1)

    plot_images(indexes_inliers, row_idx=0, label="INLIERS")
    plot_images(indexes_outliers, row_idx=1, label="OUTLIERS")

    # plt.tight_layout(rect=[0, 0, 1, 0.93])
    os.makedirs(os.path.dirname(path_inliers_outliers), exist_ok=True)
    plt.savefig(path_inliers_outliers)
    plt.close()


def compute_avg_face_embedd_iteratively(embedds, thresh=0.5):
    selected_idxs = np.array([0], dtype=int)
    avg_embedd = embedds[selected_idxs]
    for idx_candidate_embedd in range(1, len(embedds)):
        cossim = cosine_similarity(embedds[idx_candidate_embedd], avg_embedd)
        if cossim >= thresh:
            selected_idxs = np.append(selected_idxs, idx_candidate_embedd)
            avg_embedd = embedds[selected_idxs].mean(axis=0)
    return avg_embedd



if __name__ == '__main__':
    args = parse_args()

    args.embeddings = args.embeddings.rstrip('/')
    if not args.output_path:
        args.output_path = args.embeddings + '_OUTLIERS_INLIERS'
    os.makedirs(args.output_path, exist_ok=True)

    print(f'Searching images in \'{args.embeddings}\'')
    dict_subj_embedds_paths = get_all_files_in_path_by_subj(args.embeddings, [args.embedd_ext])
    # sys.exit(0)
    dict_subj_corresp_imgs_paths = get_all_corresponding_imgs_paths_by_subj(args.corresp_imgs, ['.jpg','.jpeg','.png'], dict_subj_embedds_paths, args.embeddings, args.embedd_ext)
    # sys.exit(0)

    print()
    total_elapsed_time = 0.0
    sim_thresholds = [0.4]
    # sim_thresholds = [0.3]
    # sim_thresholds = [0.25]
    for sim_thresh in sim_thresholds:
        args.output_path = os.path.join(args.output_path, f'thresh={sim_thresh}')
        os.makedirs(args.output_path, exist_ok=True)

        for idx_subj, (subj_name, subj_embedds_paths) in enumerate(dict_subj_embedds_paths.items()):
            if args.pattern in subj_name:
                start_time = time.time()
                print(f'{idx_subj}/{len(dict_subj_embedds_paths)} - subj_name: {subj_name}')
                
                subj_embedds = np.zeros((len(dict_subj_embedds_paths[subj_name]),512))
                corresp_imgs = np.zeros((len(dict_subj_embedds_paths[subj_name]),112,112,3))
                corresp_imgs_paths = []
                print(f'Loading embeddings and corresponding images...')
                for idx_embedd, embedd_path in enumerate(subj_embedds_paths):
                    corresp_img_path = dict_subj_corresp_imgs_paths[subj_name][idx_embedd]
                    corresp_imgs_paths.append(corresp_img_path)
                    one_img = cv2.cvtColor(cv2.imread(corresp_img_path), cv2.COLOR_BGR2RGB) / 255.0
                    corresp_imgs[idx_embedd] = one_img

                    one_embedd = torch.load(embedd_path)
                    subj_embedds[idx_embedd] = one_embedd
                # subj_avg_embedd = subj_embedds.mean(axis=0)
                subj_avg_embedd = compute_avg_face_embedd_iteratively(subj_embedds, sim_thresh)
                subj_avg_embedd = np.expand_dims(subj_avg_embedd, axis=0)
                # print('subj_avg_embedd.shape:', subj_avg_embedd.shape)
                # sys.exit(0)

                print(f'Computing similarities...')
                similarities_to_avg_embedd = np.zeros((len(dict_subj_embedds_paths[subj_name]),))
                for idx_embedd in range(len(dict_subj_embedds_paths[subj_name])):
                    similarities_to_avg_embedd[idx_embedd] = cosine_similarity(np.expand_dims(subj_embedds[idx_embedd], axis=0), subj_avg_embedd)
                # print('similarities_to_avg_embedd:', similarities_to_avg_embedd)
                # sys.exit(0)

                indexes_inliers  = np.where(similarities_to_avg_embedd >= sim_thresh)[0]
                indexes_outliers = np.where(similarities_to_avg_embedd < sim_thresh)[0]
                # print('indexes_outliers:', indexes_outliers)
                # print('indexes_inliers:', indexes_inliers)
                # sys.exit(0)
                
                title_inliers_outliers = 'Inliers and Outliers faces compared to AVG face embedding'
                filename_inliers_outliers = f'subj={subj_name}_inliers_outliers_faces_thresh={sim_thresh}_ninliers={len(indexes_inliers)}_noutliers={len(indexes_outliers)}.png'
                path_inliers_outliers = os.path.join(args.output_path, filename_inliers_outliers)
                print('Saving figure with inliers and outliers:', path_inliers_outliers)
                save_figure_with_inliers_outliers_faces(corresp_imgs_paths, corresp_imgs, similarities_to_avg_embedd,
                                                        indexes_inliers, indexes_outliers, title_inliers_outliers, path_inliers_outliers)


                elapsed_time = time.time()-start_time
                total_elapsed_time += elapsed_time
                avg_sample_time = total_elapsed_time / (idx_subj+1)
                estimated_time = avg_sample_time * (len(dict_subj_embedds_paths)-(idx_subj+1))
                print("Elapsed time: %.3fs" % elapsed_time)
                print("Avg elapsed time: %.3fs" % avg_sample_time)
                print("Total elapsed time: %.3fs,  %.3fm,  %.3fh" % (total_elapsed_time, total_elapsed_time/60, total_elapsed_time/3600))
                print("Estimated Time to Completion (ETC): %.3fs,  %.3fm,  %.3fh" % (estimated_time, estimated_time/60, estimated_time/3600))
                print('--------------')
            
            else:
                print(f'{idx_subj}/{len(dict_subj_embedds_paths)} - Skipping {subj_name}')

            # sys.exit(0)

    print('\nFinished!')
