import sys
import os
import argparse

import cv2
import numpy as np
import math
import torch
from pathlib import Path
import re
import glob
from collections import defaultdict
import shutil
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import gc

sys.path.insert(0, '..')
from backbones import get_model


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch ArcFace Training')
    parser.add_argument('--network', type=str, default='r100', help='backbone network')
    parser.add_argument('--weight', type=str, default='../trained_models/ms1mv3_arcface_r100_fp16/backbone.pth')
    # parser.add_argument('--probe', type=str, default='/experiments/adsouza/frames_DETECTED_FACES_RETINAFACE_scales=[0.5]_thresh=0.5_nms=0.4/imgs_112x112')
    parser.add_argument('--probe-selected-discarded', type=str, default='/experiments/adsouza/raw_frames_pre_selected_DETECTED_FACES_RETINAFACE_scales=[0.5]_thresh=0.5_nms=0.4/imgs_112x112_SELECTED_DISCARDED_FACES')
    parser.add_argument('--gallery', type=str, default='/experiments/adsouza/image_gallery_DETECTED_FACES_RETINAFACE_scales=[0.5]_thresh=0.5_nms=0.4/imgs_112x112')
    parser.add_argument('--output', type=str, default='')
    parser.add_argument('--thresh-to-gallery',  type=float, default=0.2)
    parser.add_argument('--thresh-to-selected', type=float, default=0.25)

    parser.add_argument('--str_begin',   default='', type=str, help='Substring to find and start processing')
    parser.add_argument('--str_end',     default='', type=str, help='Substring to find and stop processing')
    # parser.add_argument('--str_pattern', default='', type=str, help='Substring to find and stop processing')
    parser.add_argument('--dont_replace_existing_result', action='store_true')

    args = parser.parse_args()
    return args


def natural_sort(path_list):
    def convert(text):
        return int(text) if text.isdigit() else text.lower()
    def alphanum_key(path_obj):
        return [convert(c) for c in re.split(r"(\d+)", str(path_obj))]
    return sorted(path_list, key=alphanum_key)


def get_immediate_subdirs(path):
    base_path = Path(path)
    if not base_path.is_dir():
        raise ValueError(f"The path '{path}' is not a valid directory.")
    subdirs = [entry for entry in base_path.iterdir() if entry.is_dir()]
    return natural_sort(subdirs)


def load_faces_frames_videos(folder_path, file_extension=['.jpg','.png'], pattern=''):
    file_list = []
    num_files_found = 0
    for root, _, files in os.walk(folder_path):
        for filename in files:
            path_file = os.path.join(root, filename)
            for ext in file_extension:
                if pattern in path_file and path_file.lower().endswith(ext.lower()):
                    file_list.append(path_file)
                    num_files_found += 1
                    # print(num_files_found, end='\r')
    # print('')
    file_list = natural_sort(file_list)
    return file_list


def cosine_similarity(embedd1, embedd2):
    embedd1[0] /= np.linalg.norm(embedd1[0])
    embedd2[0] /= np.linalg.norm(embedd2[0])
    sim = float(np.maximum(np.dot(embedd1[0],embedd2[0])/(np.linalg.norm(embedd1[0])*np.linalg.norm(embedd2[0])), 0.0))
    return sim


def cosine_similarity_torch(gallery_norm_embedd, probe_faces_embedds):
    if len(gallery_norm_embedd.shape) == 1:
        gallery_norm_embedd = torch.unsqueeze(gallery_norm_embedd, 0)
    if len(probe_faces_embedds.shape) == 1:
        probe_faces_embedds = torch.unsqueeze(probe_faces_embedds, 0)
    gallery_norm = gallery_norm_embedd / gallery_norm_embedd.norm(dim=1, keepdim=True)
    probe_norm   = probe_faces_embedds / probe_faces_embedds.norm(dim=1, keepdim=True)
    similarity_matrix = torch.nn.functional.cosine_similarity(gallery_norm, probe_norm)
    similarity_matrix = torch.clamp(similarity_matrix, min=0.0, max=1.0)
    return similarity_matrix


@torch.no_grad()
def get_face_embedd(model, img):
    # embedd = model(img).cpu().numpy()
    embedd = model(img)
    return embedd


@torch.no_grad()
def get_face_embedd_batch(model, img, batch=32):
    # training_mode = model.training
    model.eval()
    num_samples = img.size(0)
    embeddings_list = []
    for i in range(0, num_samples, batch):
        batch_img = img[i : i + batch]
        batch_embedd = model(batch_img)
        embeddings_list.append(batch_embedd)
    if embeddings_list:
        return torch.cat(embeddings_list, dim=0)
    else:
        return torch.empty(0)


def load_trained_model(network, path_weights, device="cuda"):
    net = get_model(network, fp16=False)
    net.load_state_dict(torch.load(path_weights))
    net.eval()
    net = net.to(device)
    return net


def load_normalize_img(img, device="cuda"):
    img = cv2.imread(img)
    img = cv2.resize(img, (112, 112))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.transpose(img, (2, 0, 1))
    img = torch.from_numpy(img).unsqueeze(0).float().to(device)
    img.div_(255).sub_(0.5).div_(0.5)
    return img



def save_grid_selected_discarded_faces(
    path_gallery_img,
    selected_faces_paths, selected_faces_sims_to_gallery,
    recovered_faces_paths, recovered_faces_sims_to_gallery, recovered_faces_sims_to_selected,
    discarded_faces_paths, discarded_faces_sims_to_gallery, discarded_faces_sims_to_selected,
    title,
    output_grid_view_path,
    figsize=(20, 10),
    title_fontsize=16,
    sub_title_fontsize=14,
    wspace=0.2,
    hspace=0.3,
):
    """Generates and saves a 3-subfigure compilation layout showcasing a gallery image,

    selected faces, and discarded faces side-by-side using a uniform 10x10 grid layout.
    """

    # 1. Initialize matplotlib figure with 3 explicitly sized Mosaic Grid regions
    # Width ratios grant 20% space to the single gallery image, and 40% each to the collections
    fig, axs = plt.subplot_mosaic(
        [["gallery", "selected", "discarded"]],
        figsize=figsize,
        gridspec_kw={"width_ratios": [1, 2, 2]},
    )

    fig.suptitle(title, fontsize=title_fontsize, fontweight="bold")

    # --- PANEL 1: Single Gallery Target Face ---
    axs["gallery"].set_title("Target Gallery Face", fontsize=sub_title_fontsize)
    try:
        gallery_img = plt.imread(path_gallery_img)
        axs["gallery"].imshow(gallery_img)
    except Exception as e:
        axs["gallery"].text(
            0.5,
            0.5,
            f"Error loading image:\n{Path(path_gallery_img).name}",
            ha="center",
            va="center",
        )
    axs["gallery"].axis("off")


    def populate_selected_recovered_face_grid(ax_handle, face_paths, faces_sims,
                                              recovered_faces_paths, recovered_faces_sims_to_gallery, recovered_faces_sims_to_selected,
                                              panel_title):
        ax_handle.set_title(
            f"{panel_title} ({len(face_paths)})", fontsize=sub_title_fontsize
        )
        ax_handle.axis("off")  # Clear the outer structural layout axis boundaries

        if not face_paths:
            ax_handle.text(
                0.5, 0.5, "No Faces", ha="center", va="center", alpha=0.5
            )
            return

        # Force a uniform 10x10 grid system
        # rows, cols = 10, 10
        rows, cols = 15, 15

        # Build an internal GridSpec matrix coordinate network inside the structural bounding box
        inner_gs = ax_handle.get_subplotspec().subgridspec(
            rows, cols, wspace=wspace, hspace=hspace
        )

        # Iterate through the 10x10 grid locations (up to 100 possible slots)
        for idx in range(rows * cols):
            r, c = divmod(idx, cols)

            # If we still have face images left to draw
            if idx < len(face_paths):
                face_path = face_paths[idx]
                inner_ax = fig.add_subplot(inner_gs[r, c])
                face_sim = faces_sims[idx]

                try:
                    img = plt.imread(face_path)
                    inner_ax.imshow(img)

                    # Label with filename trimmed down for brevity
                    inner_ax.set_xlabel(
                        # Path(face_path).stem.split("_conf")[0],
                        f'{face_sim:.2f}',
                        fontsize=max(sub_title_fontsize - 6, 6),
                        labelpad=0.1
                    )
                except Exception:
                    inner_ax.text(
                        0.5, 0.5, "Err", ha="center", va="center", fontsize=6
                    )

                inner_ax.set_xticks([])
                inner_ax.set_yticks([])

                # Subtle inner borders around the populated images
                for spine in inner_ax.spines.values():
                    spine.set_color("#cccccc")
                    spine.set_linewidth(0.5)

            elif idx < len(recovered_faces_paths) + len(face_paths):
                idx_recovered = idx - len(face_paths)
                recovered_face_path = recovered_faces_paths[idx_recovered]
                inner_ax = fig.add_subplot(inner_gs[r, c])
                recovered_face_sim_to_gallery = recovered_faces_sims_to_gallery[idx_recovered].item()
                recovered_face_sim_to_selected = recovered_faces_sims_to_selected[idx_recovered].item()

                try:
                    img = plt.imread(recovered_face_path)
                    inner_ax.imshow(img)

                    # Label with filename trimmed down for brevity
                    inner_ax.set_xlabel(
                        # Path(face_path).stem.split("_conf")[0],
                        f'({recovered_face_sim_to_gallery:.2f},{recovered_face_sim_to_selected:.2f})',
                        # fontsize=max(sub_title_fontsize - 6, 6),
                        fontsize=4,
                        labelpad=0.1
                    )
                except Exception:
                    inner_ax.text(
                        0.5, 0.5, "Err", ha="center", va="center", fontsize=6
                    )

                inner_ax.set_xticks([])
                inner_ax.set_yticks([])

                # Subtle inner borders around the populated images
                for spine in inner_ax.spines.values():
                    spine.set_color("#cccccc")
                    spine.set_linewidth(0.5)


            else:
                # To keep the image sizing perfectly uniform, we generate empty placeholder
                # axes for the rest of the 10x10 grid but make them completely invisible.
                inner_ax = fig.add_subplot(inner_gs[r, c])
                inner_ax.axis("off")


    # Helper inner logic to build a fixed 10x10 grid layout for face images
    def populate_discarded_face_grid(ax_handle, face_paths, faces_sims_to_gallery, faces_sims_to_selected, panel_title):
        ax_handle.set_title(
            f"{panel_title} ({len(face_paths)})", fontsize=sub_title_fontsize
        )
        ax_handle.axis("off")  # Clear the outer structural layout axis boundaries

        if not face_paths:
            ax_handle.text(
                0.5, 0.5, "No Faces", ha="center", va="center", alpha=0.5
            )
            return

        # Force a uniform 10x10 grid system
        # rows, cols = 10, 10
        rows, cols = 15, 15

        # Build an internal GridSpec matrix coordinate network inside the structural bounding box
        inner_gs = ax_handle.get_subplotspec().subgridspec(
            rows, cols, wspace=wspace, hspace=hspace
        )

        # Iterate through the 10x10 grid locations (up to 100 possible slots)
        for idx in range(rows * cols):
            r, c = divmod(idx, cols)

            # If we still have face images left to draw
            if idx < len(face_paths):
                face_path = face_paths[idx]
                inner_ax = fig.add_subplot(inner_gs[r, c])
                face_sim_to_gallery = faces_sims_to_gallery[idx].item() if len(faces_sims_to_gallery) > 1 else 0
                face_sim_to_selected = faces_sims_to_selected[idx].item() if len(faces_sims_to_selected) > 1 else 0
                

                try:
                    img = plt.imread(face_path)
                    inner_ax.imshow(img)

                    # Label with filename trimmed down for brevity
                    inner_ax.set_xlabel(
                        # Path(face_path).stem.split("_conf")[0],
                        f'({face_sim_to_gallery:.2f},{face_sim_to_selected:.2f})',
                        # fontsize=max(sub_title_fontsize - 6, 6),
                        fontsize=4,
                        labelpad=0.1
                    )
                except Exception:
                    inner_ax.text(
                        0.5, 0.5, "Err", ha="center", va="center", fontsize=6
                    )

                inner_ax.set_xticks([])
                inner_ax.set_yticks([])

                # Subtle inner borders around the populated images
                for spine in inner_ax.spines.values():
                    spine.set_color("#cccccc")
                    spine.set_linewidth(0.5)

            else:
                # To keep the image sizing perfectly uniform, we generate empty placeholder
                # axes for the rest of the 10x10 grid but make them completely invisible.
                inner_ax = fig.add_subplot(inner_gs[r, c])
                inner_ax.axis("off")

    # --- PANEL 2: Uniform Selected Faces Sub-Grid ---
    populate_selected_recovered_face_grid(axs["selected"],
                                          selected_faces_paths, selected_faces_sims_to_gallery,
                                          recovered_faces_paths, recovered_faces_sims_to_gallery, recovered_faces_sims_to_selected,
                                          "Selected Faces")

    # --- PANEL 3: Uniform Discarded Faces Sub-Grid ---
    populate_discarded_face_grid(axs["discarded"], discarded_faces_paths, discarded_faces_sims_to_gallery, discarded_faces_sims_to_selected, "Discarded Faces")


    for pane_name, ax_handle in axs.items():
        bbox = ax_handle.get_position()
        rect = plt.Rectangle(
            (bbox.x0, bbox.y0),         # Lower-left corner coordinate
            bbox.width, bbox.height,    # Width and height of the subfigure panel
            transform=fig.transFigure,  # Bind coordinates relative to the entire figure
            fill=False,                 # Do not fill the interior background
            color="black",              # Border color (change to '#cccccc' or any hex if desired)
            linewidth=1.5,              # Thickness of the border line
            zorder=10                   # Ensures the border sits on top of everything else
        )
        fig.patches.append(rect)


    # Apply global padding configurations cleanly across margins
    # plt.tight_layout()

    # Save to disk layout destination
    save_path = Path(output_grid_view_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight", dpi=150)

    # plt.close()
    fig.clf()            # Clear the figure content
    plt.close(fig)       # Close the specific figure window
    del fig, axs         # Delete the Python references
    gc.collect()         # Force Python's garbage collector to run



def copy_selected_discarded_frames(
    selected_faces_paths,
    discarded_faces_paths,
    selected_target_dir,
    discarded_target_dir,
):
    # 2. Define and create the 'selected' subfolder
    # selected_target_dir = base_output_path / "selected"
    selected_target_dir = Path(selected_target_dir)
    selected_target_dir.mkdir(parents=True, exist_ok=True)

    # 3. Define and create the 'discarded' subfolder
    # discarded_target_dir = base_output_path / "discarded"
    discarded_target_dir = Path(discarded_target_dir)
    discarded_target_dir.mkdir(parents=True, exist_ok=True)

    # 4. Copy the selected files
    # print(f"Copying {len(selected_faces_paths)} files to {selected_target_dir}...")
    if not selected_faces_paths is None:
        for file_path in selected_faces_paths:
            path_obj = Path(file_path)
            if path_obj.is_file():
                # shutil.copy2 preserves original file metadata (timestamps, etc.)
                shutil.copy2(path_obj, selected_target_dir / path_obj.name)
            else:
                print(f"Warning: File not found, skipping: {file_path}")

    # 5. Copy the discarded files
    # print(f"Copying {len(discarded_faces_paths)} files to {discarded_target_dir}...")
    if not discarded_faces_paths is None:
        for file_path in discarded_faces_paths:
            path_obj = Path(file_path)
            if path_obj.is_file():
                shutil.copy2(path_obj, discarded_target_dir / path_obj.name)
            else:
                print(f"Warning: File not found, skipping: {file_path}")

    # print("File organization completed successfully!")


def save_list_to_text_file(list_str, path_file):
    try:
        with open(path_file, "w", encoding="utf-8") as file:
            for item in list_str:
                file.write(f"{item}\n")
        # print(f"Successfully saved {len(list_str)} items to '{path_file}'.")
    except OSError as e:
        print(f"An error occurred while writing to the file: {e}")



if __name__ == "__main__":
    args = parse_args()

    if not args.output:
        args.output = f"{args.probe_selected_discarded}_SELECTION_2_thresh-to-gallery={args.thresh_to_gallery}_thresh-to-selected={args.thresh_to_selected}"
    
    list_videos_with_recovered_faces = []
    path_list_videos_with_recovered_faces = f'{args.output}_videos_with_recovered_faces.txt'

    print(f'Loading trained model ({args.network}): \'{args.weight}\'')
    model = load_trained_model(args.network, args.weight)


    print(f'Loading probe subjects paths: \'{args.probe_selected_discarded}\'')
    all_probe_subj_paths = get_immediate_subdirs(args.probe_selected_discarded)    # /experiments/adsouza/raw_frames_pre_selected_DETECTED_FACES_RETINAFACE_scales=[0.5]_thresh=0.5_nms=0.4/imgs_112x112_SELECTED_DISCARDED_FACES/6
    # print('    all_probe_subj_paths:', all_probe_subj_paths)
    print('    len(all_probe_subj_paths):', len(all_probe_subj_paths))
    # sys.exit(0)

    begin_index_str = 0
    end_index_str = len(all_probe_subj_paths)

    if args.str_begin != '':
        print('\nSearching str_begin \'' + args.str_begin + '\' ...  ')
        for i, probe_subj_path in enumerate(all_probe_subj_paths):
            # print('str(probe_subj_path):', str(probe_subj_path))
            if args.str_begin in str(probe_subj_path):
                begin_index_str = i
                print('    found at', begin_index_str)
                break

    if args.str_end != '':
        print('\nSearching str_end \'' + args.str_end + '\' ...  ')
        for i, probe_subj_path in enumerate(all_probe_subj_paths):
            # print('str(probe_subj_path):', str(probe_subj_path))
            if args.str_end in str(probe_subj_path):
                end_index_str = i
                print('    found at', end_index_str)
                break
    
    print('\n------------------------')
    print('begin_index_str:', begin_index_str)
    print('end_index_str:', end_index_str)
    print('------------------------\n')
    # sys.exit(0)

    total_num_videos_with_recovered_faces = 0
    total_num_recovered_faces = 0

    total_exec_time = 0.0

    for idx_probe_subj, probe_subj_path in enumerate(all_probe_subj_paths):
        subj_start_time = time.time()
        print("-----------------------")
        print(f"Subj {idx_probe_subj}/{len(all_probe_subj_paths)} - '{probe_subj_path}'")
        subj_name = os.path.basename(probe_subj_path)

        if idx_probe_subj >= begin_index_str and idx_probe_subj <= end_index_str:
            all_probe_subj_videos_paths = get_immediate_subdirs(probe_subj_path)    # /experiments/adsouza/raw_frames_pre_selected_DETECTED_FACES_RETINAFACE_scales=[0.5]_thresh=0.5_nms=0.4/imgs_112x112_SELECTED_DISCARDED_FACES/6/elQqvUdVUSY
            if len(all_probe_subj_videos_paths) > 0:
                pattern_gallery_img = os.path.join(args.gallery, f'{subj_name}_*.png').replace('[','*')
                print('    pattern_gallery_img:', pattern_gallery_img)
                path_gallery_img = glob.glob(pattern_gallery_img)
                if len(path_gallery_img) == 0:
                    print(f"    Skipping subj \'{subj_name}\'. Gallery image not found with pattern \'{pattern_gallery_img}\'")
                    continue
                path_gallery_img = path_gallery_img[0]
                # print('path_gallery_img:', path_gallery_img)
                gallery_norm_img = load_normalize_img(path_gallery_img)
                gallery_norm_embedd = get_face_embedd(model, gallery_norm_img)
                

                for idx_probe_video, path_probe_video_path in enumerate(all_probe_subj_videos_paths):
                    video_start_time = time.time()
                    print('    ---------')
                    print(f"    Video {idx_probe_video}/{len(all_probe_subj_videos_paths)} - '{path_probe_video_path}'")
                    video_name = os.path.basename(path_probe_video_path)
                    output_selected_discarded_frames_dir = os.path.join(f"{args.output}", subj_name, video_name)
                    selected_discarded_faces_figure_filename = f'selected_discarded_faces_subj={subj_name}_video={video_name}_SELECTION_2.png'
                    output_grid_view_path = os.path.join(f"{args.output}_GRIDS_VIEWS", subj_name, selected_discarded_faces_figure_filename)

                    path_probe_video_selected_faces  = os.path.join(path_probe_video_path, "selected")
                    path_probe_video_discarded_faces = os.path.join(path_probe_video_path, "discarded")
                    print(f'        path_probe_video_selected_faces:', path_probe_video_selected_faces)
                    print(f'        path_probe_video_discarded_faces:', path_probe_video_discarded_faces)

                    paths_probe_video_selected_faces  = load_faces_frames_videos(path_probe_video_selected_faces)
                    paths_probe_video_discarded_faces = load_faces_frames_videos(path_probe_video_discarded_faces)
                    print('            len(paths_probe_video_selected_faces):', len(paths_probe_video_selected_faces))
                    print('            len(paths_probe_video_discarded_faces):', len(paths_probe_video_discarded_faces))
                    # sys.exit(0)


                    if args.dont_replace_existing_result:
                        if os.path.isdir(output_selected_discarded_frames_dir) and os.path.isfile(output_grid_view_path):
                            img_grid_view = cv2.imread(output_grid_view_path)
                            if not img_grid_view is None:
                                print(f"        Skipping video already processed!")
                                continue


                    recovery_faces_paths            = []
                    recovery_faces_sims_to_gallery  = []
                    recovery_faces_sims_to_selected = []

                    discarded_faces_paths            = []
                    discarded_faces_sims_to_gallery  = []
                    discarded_faces_sims_to_selected = []

                    selected_faces_to_gallery_cossim = []

                    if len(paths_probe_video_selected_faces) > 0 and len(paths_probe_video_discarded_faces) > 0:
                        probe_video_selected_faces_norm_img  = torch.cat([load_normalize_img(path_selected_frame) for path_selected_frame in paths_probe_video_selected_faces])
                        probe_video_discarded_faces_norm_img = torch.cat([load_normalize_img(path_discarded_frame) for path_discarded_frame in paths_probe_video_discarded_faces])

                        probe_video_selected_faces_embedds  = get_face_embedd_batch(model, probe_video_selected_faces_norm_img)
                        probe_video_discarded_faces_embedds = get_face_embedd_batch(model, probe_video_discarded_faces_norm_img)
                        selected_faces_to_gallery_cossim  = cosine_similarity_torch(probe_video_selected_faces_embedds, gallery_norm_embedd).cpu().numpy()


                        for idx_discarded_face_embedd, (path_discarded_face, discarded_face_embedd) in enumerate(zip(paths_probe_video_discarded_faces, probe_video_discarded_faces_embedds)):
                            print(f"        checking face {idx_discarded_face_embedd}/{len(paths_probe_video_discarded_faces)} - '{path_discarded_face}'", end='\r')
                            discarded_face_to_gallery_cossim  = cosine_similarity_torch(discarded_face_embedd, gallery_norm_embedd).cpu().numpy()
                            discarded_face_to_selected_cossim = cosine_similarity_torch(discarded_face_embedd, probe_video_selected_faces_embedds).cpu().numpy()
                            
                            if discarded_face_to_gallery_cossim >= args.thresh_to_gallery and discarded_face_to_selected_cossim.mean() >= args.thresh_to_selected:
                                recovery_faces_paths.append(path_discarded_face)
                                recovery_faces_sims_to_gallery.append(discarded_face_to_gallery_cossim)
                                recovery_faces_sims_to_selected.append(discarded_face_to_selected_cossim.mean())
                            else:
                                discarded_faces_paths.append(path_discarded_face)
                                discarded_faces_sims_to_gallery.append(discarded_face_to_gallery_cossim)
                                discarded_faces_sims_to_selected.append(discarded_face_to_selected_cossim.mean())

                        print()
                        print(f'        len(paths_probe_video_selected_faces): {len(paths_probe_video_selected_faces)}')
                        print(f'        len(recovery_faces_paths):             {len(recovery_faces_paths)}')
                        print(f'        len(discarded_faces_paths):            {len(discarded_faces_paths)}')
                        assert len(paths_probe_video_selected_faces) + len(recovery_faces_paths) + len(discarded_faces_paths) == len(paths_probe_video_selected_faces) + len(paths_probe_video_discarded_faces)


                    elif len(paths_probe_video_selected_faces) > 0:
                        probe_video_selected_faces_norm_img  = torch.cat([load_normalize_img(path_selected_frame) for path_selected_frame in paths_probe_video_selected_faces])
                        probe_video_selected_faces_embedds  = get_face_embedd_batch(model, probe_video_selected_faces_norm_img)
                        selected_faces_to_gallery_cossim  = cosine_similarity_torch(probe_video_selected_faces_embedds, gallery_norm_embedd).cpu().numpy()


                    elif len(paths_probe_video_discarded_faces) > 0:
                        discarded_faces_paths = paths_probe_video_discarded_faces
                        probe_video_discarded_faces_norm_img = torch.cat([load_normalize_img(path_discarded_frame) for path_discarded_frame in paths_probe_video_discarded_faces])
                        probe_video_discarded_faces_embedds = get_face_embedd_batch(model, probe_video_discarded_faces_norm_img)
                        discarded_faces_sims_to_gallery  = cosine_similarity_torch(probe_video_discarded_faces_embedds, gallery_norm_embedd).cpu().numpy()
                        

                    output_selected_faces_dir  = os.path.join(output_selected_discarded_frames_dir, "selected")
                    output_discarded_faces_dir = os.path.join(output_selected_discarded_frames_dir, "discarded")
                    os.makedirs(output_selected_faces_dir, exist_ok=True)
                    os.makedirs(output_discarded_faces_dir, exist_ok=True)

                    if len(paths_probe_video_selected_faces) > 0:
                        print(f'        Copying initially selected faces: \'{output_selected_discarded_frames_dir}\'')
                        copy_selected_discarded_frames(paths_probe_video_selected_faces, None, output_selected_faces_dir, output_discarded_faces_dir)    
                    if len(recovery_faces_paths) > 0:
                        total_num_videos_with_recovered_faces += 1
                        total_num_recovered_faces += len(recovery_faces_paths)
                        list_videos_with_recovered_faces.append(path_probe_video_path)
                        print(f'        Saving list of videos with recovered faces: \'{path_list_videos_with_recovered_faces}\'')
                        save_list_to_text_file(list_videos_with_recovered_faces, path_list_videos_with_recovered_faces)
                        print(f'        Copying recovered faces: \'{output_selected_discarded_frames_dir}\'')
                        copy_selected_discarded_frames(recovery_faces_paths, None, output_selected_faces_dir, output_discarded_faces_dir)
                    if len(discarded_faces_paths) > 0:
                        print(f'        Copying discarded faces: \'{output_selected_discarded_frames_dir}\'')
                        copy_selected_discarded_frames(None, discarded_faces_paths, output_selected_faces_dir, output_discarded_faces_dir)
                    # sys.exit(0)                    

                    os.makedirs(os.path.dirname(output_grid_view_path), exist_ok=True)
                    title = f"Subj: '{subj_name}'    Video: '{video_name}'"
                    print(f'        Saving grid of selected and discarded faces: \'{output_grid_view_path}\'')
                    save_grid_selected_discarded_faces(path_gallery_img,
                                                       paths_probe_video_selected_faces, selected_faces_to_gallery_cossim,
                                                       recovery_faces_paths, recovery_faces_sims_to_gallery, recovery_faces_sims_to_selected,
                                                       discarded_faces_paths, discarded_faces_sims_to_gallery, discarded_faces_sims_to_selected,
                                                       title, output_grid_view_path)
                    
                    print(f'    total_num_videos_with_recovered_faces:', total_num_videos_with_recovered_faces)
                    print(f'    total_num_recovered_faces:            ', total_num_recovered_faces)
                    video_end_time = time.time()
                    video_time = video_end_time - video_start_time
                    print(f"        video_time:       {video_time:.2f}s    {video_time/60:.2f}m    {video_time/3600:.2f}h")
        
                    
                    # sys.exit(0)    # end video
            # sys.exit(0)    # end subj
            # print("-----------------------")
        
        else:
            print(f"    Skipping subj \'{subj_name}\' due to index constraints (begin_index_str: {begin_index_str}, end_index_str: {end_index_str})")
    
        subj_end_time = time.time()
        subj_time = subj_end_time - subj_start_time
        total_exec_time += subj_time
        remain_time = (len(all_probe_subj_paths)-idx_probe_subj+1) * subj_time

        print(f"        subj_time:       {subj_time:.2f}s    {subj_time/60:.2f}m    {subj_time/3600:.2f}h")
        print(f"        total_exec_time: {total_exec_time:.2f}s    {total_exec_time/60:.2f}m    {total_exec_time/3600:.2f}h")
        print(f"        remain_time:     {remain_time:.2f}s    {remain_time/60:.2f}m    {remain_time/3600:.2f}h")

    print('\nFinished!\n')
