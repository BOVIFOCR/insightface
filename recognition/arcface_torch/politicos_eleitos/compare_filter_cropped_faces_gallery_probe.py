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
import matplotlib.pyplot as plt

sys.path.insert(0, '..')
from backbones import get_model


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch ArcFace Training')
    parser.add_argument('--network', type=str, default='r100', help='backbone network')
    parser.add_argument('--weight', type=str, default='../trained_models/ms1mv3_arcface_r100_fp16/backbone.pth')
    parser.add_argument('--probe', type=str, default='/experiments/adsouza/frames_DETECTED_FACES_RETINAFACE_scales=[0.5]_thresh=0.5_nms=0.4/imgs_112x112')
    parser.add_argument('--gallery', type=str, default='/experiments/adsouza/image_gallery_DETECTED_FACES_RETINAFACE_scales=[0.5]_thresh=0.5_nms=0.4/imgs_112x112')
    parser.add_argument('--output', type=str, default='')
    parser.add_argument('--thresh', type=float, default=0.3)

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


def load_faces_frames_videos(videos_path):
    base_path = Path(videos_path)
    if not base_path.is_dir():
        raise ValueError(f"The path '{videos_path}' is not a valid directory.")

    frames_dict = defaultdict(list)
    # frame_prefix_pattern = re.compile(r"^(frame_\d+)")
    frame_prefix_pattern = re.compile(r"^(0*\d+)")
    for entry in base_path.iterdir():
        if entry.is_file():
            filename = entry.name
            match = frame_prefix_pattern.match(filename)
            if match:
                frame_key = match.group(1)
                frames_dict[frame_key].append(os.path.join(base_path,filename))

    for frame_key in frames_dict:
        frames_dict[frame_key] = natural_sort(frames_dict[frame_key])

    sorted_keys = natural_sort(list(frames_dict.keys()))
    return {key: frames_dict[key] for key in sorted_keys}


def cosine_similarity(embedd1, embedd2):
    embedd1[0] /= np.linalg.norm(embedd1[0])
    embedd2[0] /= np.linalg.norm(embedd2[0])
    sim = float(np.maximum(np.dot(embedd1[0],embedd2[0])/(np.linalg.norm(embedd1[0])*np.linalg.norm(embedd2[0])), 0.0))
    return sim


def cosine_similarity_torch(gallery_norm_embedd, probe_faces_embedds):
    gallery_norm = gallery_norm_embedd / gallery_norm_embedd.norm(dim=1, keepdim=True)
    probe_norm   = probe_faces_embedds / probe_faces_embedds.norm(dim=1, keepdim=True)
    similarity_matrix = torch.nn.functional.cosine_similarity(gallery_norm, probe_norm)
    return similarity_matrix


@torch.no_grad()
def get_face_embedd(model, img):
    # embedd = model(img).cpu().numpy()
    embedd = model(img)
    return embedd


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
    selected_faces_paths,
    selected_faces_sims,
    discarded_faces_paths,
    discarded_faces_sims,
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

    # Helper inner logic to build a fixed 10x10 grid layout for face images
    def populate_fixed_face_grid(ax_handle, face_paths, faces_sims, panel_title):
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

            else:
                # To keep the image sizing perfectly uniform, we generate empty placeholder
                # axes for the rest of the 10x10 grid but make them completely invisible.
                inner_ax = fig.add_subplot(inner_gs[r, c])
                inner_ax.axis("off")

    # --- PANEL 2: Uniform Selected Faces Sub-Grid ---
    populate_fixed_face_grid(axs["selected"], selected_faces_paths, selected_faces_sims, "Selected Faces")

    # --- PANEL 3: Uniform Discarded Faces Sub-Grid ---
    populate_fixed_face_grid(axs["discarded"], discarded_faces_paths, discarded_faces_sims, "Discarded Faces")


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
    plt.close()



def copy_selected_discarded_frames(
    selected_faces_paths,
    discarded_faces_paths,
    output_selected_discarded_frames_dir,
):
    base_output_path = Path(output_selected_discarded_frames_dir)

    # 2. Define and create the 'selected' subfolder
    selected_target_dir = base_output_path / "selected"
    selected_target_dir.mkdir(parents=True, exist_ok=True)

    # 3. Define and create the 'discarded' subfolder
    discarded_target_dir = base_output_path / "discarded"
    discarded_target_dir.mkdir(parents=True, exist_ok=True)

    # 4. Copy the selected files
    # print(f"Copying {len(selected_faces_paths)} files to {selected_target_dir}...")
    for file_path in selected_faces_paths:
        path_obj = Path(file_path)
        if path_obj.is_file():
            # shutil.copy2 preserves original file metadata (timestamps, etc.)
            shutil.copy2(path_obj, selected_target_dir / path_obj.name)
        else:
            print(f"Warning: File not found, skipping: {file_path}")

    # 5. Copy the discarded files
    # print(f"Copying {len(discarded_faces_paths)} files to {discarded_target_dir}...")
    for file_path in discarded_faces_paths:
        path_obj = Path(file_path)
        if path_obj.is_file():
            shutil.copy2(path_obj, discarded_target_dir / path_obj.name)
        else:
            print(f"Warning: File not found, skipping: {file_path}")

    # print("File organization completed successfully!")





if __name__ == "__main__":
    args = parse_args()

    if not args.output:
        args.output = f"{args.probe}_SELECTED_DISCARDED_FACES"

    print(f'Loading trained model ({args.network}): \'{args.weight}\'')
    model = load_trained_model(args.network, args.weight)


    print(f'Loading probe subjects paths: \'{args.probe}\'')
    all_probe_subj_paths = get_immediate_subdirs(args.probe)
    # print('    all_probe_subj_paths:', all_probe_subj_paths)
    print('    len(all_probe_subj_paths):', len(all_probe_subj_paths))
    # sys.exit(0)

    begin_index_str = 0
    end_index_str = len(all_probe_subj_paths)

    if args.str_begin != '':
        print('\nSearching str_begin \'' + args.str_begin + '\' ...  ')
        for i, probe_subj_path in enumerate(all_probe_subj_paths):
            if args.str_begin in str(probe_subj_path):
                begin_index_str = i
                print('    found at', begin_index_str)
                break

    if args.str_end != '':
        print('\nSearching str_end \'' + args.str_end + '\' ...  ')
        for i, probe_subj_path in enumerate(all_probe_subj_paths):
            if args.str_end in str(probe_subj_path):
                end_index_str = i
                print('    found at', end_index_str)
                break
    
    print('\n------------------------')
    print('begin_index_str:', begin_index_str)
    print('end_index_str:', end_index_str)
    print('------------------------\n')
    # sys.exit(0)

    for idx_probe_subj, probe_subj_path in enumerate(all_probe_subj_paths):
        print("-----------------------")
        print(f"Subj {idx_probe_subj}/{len(all_probe_subj_paths)} - '{probe_subj_path}'")

        if idx_probe_subj >= begin_index_str and idx_probe_subj <= end_index_str:
            all_probe_subj_videos_paths = get_immediate_subdirs(probe_subj_path)
            if len(all_probe_subj_videos_paths) > 0:
                subj_name = os.path.basename(probe_subj_path)
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
                
                for idx_probe_video, path_probe_videos_path in enumerate(all_probe_subj_videos_paths):
                    print(f"    Video {idx_probe_video}/{len(all_probe_subj_videos_paths)} - '{path_probe_videos_path}'")
                    video_name = os.path.basename(path_probe_videos_path)
                    dict_frames_probe_video = load_faces_frames_videos(path_probe_videos_path)
                    # print('dict_frames_probe_video:', dict_frames_probe_video)

                    output_selected_discarded_frames_dir = os.path.join(f"{args.output}", subj_name, video_name)
                    selected_discarded_faces_figure_filename = f'selected_discarded_faces_subj={subj_name}_video={video_name}.png'
                    output_grid_view_path = os.path.join(f"{args.output}_GRIDS_VIEWS", subj_name, selected_discarded_faces_figure_filename)

                    if args.dont_replace_existing_result:
                        if os.path.isdir(output_selected_discarded_frames_dir) and os.path.isfile(output_grid_view_path):
                            img_grid_view = cv2.imread(output_grid_view_path)
                            if not img_grid_view is None:
                                print(f"        Skipping video already processed!")
                                continue


                    selected_faces_paths  = []
                    selected_faces_sims   = []

                    discarded_faces_paths = []
                    discarded_faces_sims  = []
                    for idx_frame_key, frame_key in enumerate(list(dict_frames_probe_video.keys())):
                        print(f"        Frame {idx_frame_key}/{frame_key}")
                        probe_faces_paths = dict_frames_probe_video[frame_key]
                        assert len(probe_faces_paths) > 0
                        probe_faces_imgs = [load_normalize_img(probe_face_path) for probe_face_path in probe_faces_paths]
                        # print('probe_faces_imgs[0].shape:', probe_faces_imgs[0].shape)
                        probe_faces_imgs = torch.cat(probe_faces_imgs).to(device="cuda")
                        probe_faces_embedds = get_face_embedd(model, probe_faces_imgs)
                        # print('probe_faces_embedds.shape:', probe_faces_embedds.shape)

                        # probe_cossims = np.array([cosine_similarity(gallery_norm_embedd, probe_faces_embedd) for probe_faces_embedd in probe_faces_embedds])
                        probe_cossims = cosine_similarity_torch(gallery_norm_embedd, probe_faces_embedds)
                        probe_cossims = probe_cossims.cpu().numpy()
                        print('            probe_cossims:', probe_cossims)
                        if probe_cossims.max() >= args.thresh:
                            index_max_sim = np.where(probe_cossims == probe_cossims.max())[0].item()
                            selected_faces_paths.append(probe_faces_paths[index_max_sim])
                            selected_faces_sims.append(probe_cossims[index_max_sim])
                            # print(f'            index_max_sim: {index_max_sim} - selected_face_path:', selected_face_path)
                            print(f'            index_max_sim: {index_max_sim}')
                            for i in range(len(probe_cossims)):
                                if i != index_max_sim: 
                                    discarded_faces_paths.append(probe_faces_paths[i])
                                    discarded_faces_sims.append(probe_cossims[i])
                        else:
                            discarded_faces_paths.extend(probe_faces_paths)
                            discarded_faces_sims.extend(list(probe_cossims))

                    print(f'        len(selected_faces_paths):  {len(selected_faces_paths)}')
                    print(f'        len(discarded_faces_paths): {len(discarded_faces_paths)}')


                    # output_selected_discarded_frames_dir = os.path.join(f"{args.output}", subj_name, video_name)
                    os.makedirs(output_selected_discarded_frames_dir, exist_ok=True)
                    print(f'        Copying selected and discarded faces: \'{output_selected_discarded_frames_dir}\'')
                    copy_selected_discarded_frames(selected_faces_paths, discarded_faces_paths, output_selected_discarded_frames_dir)
                    

                    # selected_discarded_faces_figure_filename = f'selected_discarded_faces_subj={subj_name}_video={video_name}.png'
                    # output_grid_view_path = os.path.join(f"{args.output}_GRIDS_VIEWS", subj_name, selected_discarded_faces_figure_filename)
                    os.makedirs(os.path.dirname(output_grid_view_path), exist_ok=True)
                    title = f"Subj: '{subj_name}'    Video: '{video_name}'"
                    print(f'        Saving grid of selected and discarded faces: \'{output_grid_view_path}\'')
                    save_grid_selected_discarded_faces(path_gallery_img,
                                                    selected_faces_paths, selected_faces_sims,
                                                    discarded_faces_paths, discarded_faces_sims,
                                                    title, output_grid_view_path)

                # sys.exit(0)
            # sys.exit(0)
            # print("-----------------------")
        else:
            print(f"    Skipping subj \'{subj_name}\' due to index constraints (begin_index_str: {begin_index_str}, end_index_str: {end_index_str})")
            


    sys.exit(0)

    print(f'Loading and normalizing images {args.img1}, {args.img2}')
    norm_img1 = load_normalize_img(args.img1)
    norm_img2 = load_normalize_img(args.img2)

    print(f'Computing face embeddings')
    face_embedd1 = get_face_embedd(model, norm_img1)
    face_embedd2 = get_face_embedd(model, norm_img2)

    print(f'Computing cosine similarity (0: lowest, 1: highest)')
    sim = cosine_similarity(face_embedd1, face_embedd2)
    print(f'Cosine similarity: {sim}')

    if sim >= args.thresh:
        print('    SAME PERSON')
    else:
        print('    DIFFERENT PERSON')
