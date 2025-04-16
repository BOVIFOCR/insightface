# Based on https://github.com/jacobgil/pytorch-grad-cam/blob/master/tutorials/Pixel%20Attribution%20for%20embeddings.ipynb

import os, sys
import argparse

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

import matplotlib.pyplot as plt
import matplotlib.cm as cm

from backbones import get_model

from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.image import preprocess_image, show_cam_on_image


def cosine_similarity(embedd1, embedd2):
    embedd1[0] /= np.linalg.norm(embedd1[0])
    embedd2[0] /= np.linalg.norm(embedd2[0])
    sim = float(np.maximum(np.dot(embedd1[0],embedd2[0])/(np.linalg.norm(embedd1[0])*np.linalg.norm(embedd2[0])), 0.0))
    return sim


@torch.no_grad()
def get_face_embedd(model, img):
    embedd = model(img).numpy()
    return embedd


def load_trained_model(network, path_weights):
    net = get_model(network, fp16=False)
    net.load_state_dict(torch.load(path_weights))
    net.eval()
    return net


def load_normalize_img(img):
    img = cv2.imread(img)
    img = cv2.resize(img, (112, 112))

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.transpose(img, (2, 0, 1))
    img = torch.from_numpy(img).unsqueeze(0).float()
    img.div_(255).sub_(0.5).div_(0.5)
    return img



def get_similarity_score(feature_map, reference_embedding):
    # Normalize embeddings
    feature_map = F.normalize(feature_map, p=2, dim=1)
    reference_embedding = F.normalize(reference_embedding, p=2, dim=1)
    # Compute cosine similarity (proxy for classification score)
    return (feature_map * reference_embedding).sum(dim=1, keepdim=True)


def plot_face_verification_heatmaps(title, save_path, image1, image2, heatmaps1, heatmaps2, heatmap_titles=None):
    assert len(heatmaps1) == len(heatmaps2), "Both images must have the same number of heatmaps"
    N = len(heatmaps1) + 1  # +1 for the original image column

    fig, axes = plt.subplots(2, N, figsize=(1.8*N, 4))
    fig.suptitle(title, fontsize=16)

    axes[0, 0].imshow(image1)
    axes[0, 0].set_title("Original Images")
    axes[0, 0].axis('off')

    axes[1, 0].imshow(image2)
    # axes[1, 0].set_title("Original Image 2")
    axes[1, 0].axis('off')

    for i in range(len(heatmaps1)):
        axes[0, i + 1].imshow(heatmaps1[i], cmap=cm.jet)
        axes[0, i + 1].set_title(heatmap_titles[i] if heatmap_titles else f"Heatmap {i + 1}")
        axes[0, i + 1].axis('off')

        axes[1, i + 1].imshow(heatmaps2[i], cmap=cm.jet)
        # axes[1, i + 1].set_title(heatmap_titles[i] if heatmap_titles else f"Heatmap {i + 1}")
        axes[1, i + 1].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch ArcFace Training')
    parser.add_argument('--network', type=str, default='r100', help='backbone network')
    parser.add_argument('--weight', type=str, default='./trained_models/ms1mv3_arcface_r100_fp16/backbone.pth')
    parser.add_argument('--img1', type=str, default='Aaron_Peirsol_0001.png')
    parser.add_argument('--img2', type=str, default='Aaron_Peirsol_0002.png')
    parser.add_argument('--thresh', type=float, default=0.5)
    parser.add_argument('--pair-label', type=str)
    args = parser.parse_args()
    args.pair_label = args.pair_label.lower() in ['true', '1', 't', 'y', 'yes']

    print(f'Loading trained model ({args.network}): {args.weight}')
    model = load_trained_model(args.network, args.weight)

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



    
    # Grad-CAM
    class GenuinePairEmbeddingTarget:
        def __init__(self, reference_embedding):
            self.reference_embedding = reference_embedding

        def __call__(self, target_embedd):
            target_embedd = torch.squeeze(target_embedd)
            reference_embedding = torch.squeeze(self.reference_embedding)
            cos = torch.nn.CosineSimilarity(dim=0)
            similarity_score = cos(reference_embedding, target_embedd)
            # print('GenuinePairEmbeddingTarget - similarity_score:', similarity_score)
            return similarity_score

    class ImpostorPairEmbeddingTarget:
        def __init__(self, reference_embedding):
            self.reference_embedding = reference_embedding

        def __call__(self, target_embedd):
            target_embedd = torch.squeeze(target_embedd)
            reference_embedding = torch.squeeze(self.reference_embedding)
            cos = torch.nn.CosineSimilarity(dim=0)
            similarity_score = cos(reference_embedding, target_embedd)
            # print('ImpostorPairEmbeddingTarget - similarity_score:', similarity_score)
            return 1.0 - similarity_score


    input_tensor1 = norm_img1
    input_tensor2 = norm_img2

    print('input_tensor.shape:', input_tensor1.shape)
    features1 = model(input_tensor1)
    features2 = model(input_tensor2)

    if args.pair_label:
        # print('GenuinePairEmbeddingTarget')
        target1 = GenuinePairEmbeddingTarget(features1)
        target2 = GenuinePairEmbeddingTarget(features2)
    else:
        # print('ImpostorPairEmbeddingTarget')
        target1 = ImpostorPairEmbeddingTarget(features1)
        target2 = ImpostorPairEmbeddingTarget(features2)


    camClasses = [AblationCAM, EigenCAM, FullGrad, GradCAM, GradCAMPlusPlus, HiResCAM, ScoreCAM, XGradCAM]
    
    list_face_heatmap1 = []
    list_face_heatmap2 = []

    stacked_image1 = cv2.imread(args.img1)
    stacked_image2 = cv2.imread(args.img2)

    for idx_camClass, camClass in enumerate(camClasses):

        # for layer_idx in range(-1, -4, -1):
        for layer_idx in range(-1, -2, -1):
            target_layers = [model.layer4[layer_idx]]

            cam1 = camClass(model=model, target_layers=target_layers)
            cam2 = camClass(model=model, target_layers=target_layers)
            print(f"{idx_camClass}/{len(camClasses)} - {camClass.__name__}    layer4[{layer_idx}]")

            grayscale_cam1 = cam1(input_tensor1, targets=[target2])
            grayscale_cam2 = cam2(input_tensor2, targets=[target1])

            image1 = cv2.imread(args.img1) / 255.0
            heatmap1 = show_cam_on_image(image1, grayscale_cam1[0], use_rgb=True)
            list_face_heatmap1.append(heatmap1)

            image2 = cv2.imread(args.img2) / 255.0
            heatmap2 = show_cam_on_image(image2, grayscale_cam2[0], use_rgb=True)
            list_face_heatmap2.append(heatmap2)

    image1 = cv2.cvtColor(cv2.imread(args.img1), cv2.COLOR_BGR2RGB)
    image2 = cv2.cvtColor(cv2.imread(args.img2), cv2.COLOR_BGR2RGB)
    title = 'Face Verification Heatmaps'
    save_path = f"grad_cam_results_pair-label={args.pair_label}_layer4[{layer_idx}]_img1={os.path.basename(args.img1)}_img2={os.path.basename(args.img2)}.png"
    heatmap_titles = [camClasse.__name__ for camClasse in camClasses]
    print(f"Saving results at: {save_path}")
    plot_face_verification_heatmaps(title, save_path, image1, image2, list_face_heatmap1, list_face_heatmap2, heatmap_titles)