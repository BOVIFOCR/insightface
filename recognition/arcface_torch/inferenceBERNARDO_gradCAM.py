import argparse

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from backbones import get_model

from pytorch_grad_cam import GradCAM
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



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch ArcFace Training')
    parser.add_argument('--network', type=str, default='r100', help='backbone network')
    parser.add_argument('--weight', type=str, default='./trained_models/ms1mv3_arcface_r100_fp16/backbone.pth')
    parser.add_argument('--img1', type=str, default='Aaron_Peirsol_0001.png')
    parser.add_argument('--img2', type=str, default='Aaron_Peirsol_0002.png')
    parser.add_argument('--thresh', type=float, default=0.5)
    args = parser.parse_args()

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
    class EmbeddingTarget:
        def __init__(self, reference_embedding):
            self.reference_embedding = reference_embedding  # Precomputed reference face embedding

        def __call__(self, model_output):
            if len(model_output.shape) == 1:
                model_output = model_output.unsqueeze(0)
            
            feature_map = F.normalize(model_output, p=2, dim=1)  
            reference_embedding = F.normalize(self.reference_embedding, p=2, dim=1)  

            similarity_score = (feature_map * reference_embedding).sum(dim=1, keepdim=True)
            print('similarity_score:', similarity_score)
            return similarity_score  # Return a tensor of shape [batch_size, 1]

    for layer_idx in range(-1, -4, -1):
        target_layers = [model.layer4[layer_idx]]
        
        input_tensor1 = norm_img1
        input_tensor2 = norm_img2

        print('input_tensor.shape:', input_tensor1.shape)
        features1 = model(input_tensor1)
        features2 = model(input_tensor2)

        target1 = EmbeddingTarget(features1)
        target2 = EmbeddingTarget(features2)

        cam = GradCAM(model=model, target_layers=target_layers)
        grayscale_cam1 = cam(input_tensor1, targets=[target2])
        grayscale_cam2 = cam(input_tensor2, targets=[target1])

        image1 = cv2.imread(args.img1) / 255.0
        heatmap1 = show_cam_on_image(image1, grayscale_cam1[0], use_rgb=True)
        output_path1 = f"grad_cam_result_img1_layer4[{layer_idx}].png"
        cv2.imwrite(output_path1, cv2.cvtColor(heatmap1, cv2.COLOR_RGB2BGR))
        print(f"Grad-CAM visualization saved at: {output_path1}")

        image2 = cv2.imread(args.img2) / 255.0
        heatmap2 = show_cam_on_image(image2, grayscale_cam2[0], use_rgb=True)
        output_path2 = f"grad_cam_result_img2_layer4[{layer_idx}].png"
        cv2.imwrite(output_path2, cv2.cvtColor(heatmap2, cv2.COLOR_RGB2BGR))
        print(f"Grad-CAM visualization saved at: {output_path2}")