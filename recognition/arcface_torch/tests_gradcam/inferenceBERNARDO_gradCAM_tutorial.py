# Based on https://github.com/jacobgil/pytorch-grad-cam/blob/master/tutorials/Pixel%20Attribution%20for%20embeddings.ipynb

import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
from torchvision.models.segmentation import deeplabv3_resnet50
import torch
import torch.functional as F
import numpy as np
import requests
import cv2
import torchvision
from PIL import Image
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from pytorch_grad_cam import GradCAM

# A model wrapper that gets a resnet model and returns the features before the fully connected layer.
class ResnetFeatureExtractor(torch.nn.Module):
    def __init__(self, model):
        super(ResnetFeatureExtractor, self).__init__()
        self.model = model
        self.feature_extractor = torch.nn.Sequential(*list(self.model.children())[:-1])
                
    def __call__(self, x):
        # print('self.model(x).shape:', self.model(x).shape)
        # return self.model(x)
        return self.feature_extractor(x)[:, :, 0, 0]
        
resnet = torchvision.models.resnet50(pretrained=True)
resnet.eval()
model = ResnetFeatureExtractor(resnet)


def get_image_from_url(url):
    """A function that gets a URL of an image, 
    and returns a numpy image and a preprocessed
    torch tensor ready to pass to the model """

    if '://' in url:
        img = np.array(Image.open(requests.get(url, stream=True).raw))
    else:
        img = np.array(Image.open(url))
    img = cv2.resize(img, (512, 512))
    if img.shape[2] > 3:
        img = img[:, :, :3]

    rgb_img_float = np.float32(img) / 255
    input_tensor = preprocess_image(rgb_img_float,
                                   mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
    return img, rgb_img_float, input_tensor


ref_img, ref_img_float, ref_tensor = get_image_from_url("/home/bjgbiesseck/GitHub/insightface/recognition/arcface_torch/tests_gradcam/car_reference1.png")
# ref_img, ref_img_float, ref_tensor = get_image_from_url("/home/bjgbiesseck/GitHub/insightface/recognition/arcface_torch/tests_gradcam/car_reference2.png")
# ref_img, ref_img_float, ref_tensor = get_image_from_url("/home/bjgbiesseck/GitHub/insightface/recognition/arcface_torch/tests_gradcam/car_reference3.jpg")
# ref_img, ref_img_float, ref_tensor = get_image_from_url("/home/bjgbiesseck/GitHub/insightface/recognition/arcface_torch/tests_gradcam/cloud_reference1.png")
# ref_img, ref_img_float, ref_tensor = get_image_from_url("/home/bjgbiesseck/GitHub/insightface/recognition/arcface_torch/tests_gradcam/cloud_reference2.png")
# ref_img, ref_img_float, ref_tensor = get_image_from_url("/home/bjgbiesseck/GitHub/insightface/recognition/arcface_torch/tests_gradcam/cloud_reference3.png")
ref_img_features = model(ref_tensor)[0, :]
print('ref_img_features.shape:', ref_img_features.shape)

target_img, target_img_float, target_img_tensor = get_image_from_url("/home/bjgbiesseck/GitHub/insightface/recognition/arcface_torch/tests_gradcam/car1.png")
target_img_features = model(target_img_tensor)[0, :]
print('target_img_features.shape:', target_img_features.shape)
print()


class SimilarityToConceptTarget:
    def __init__(self, features):
        self.features = features
    
    def __call__(self, model_output):
        cos = torch.nn.CosineSimilarity(dim=0)
        similarity = cos(model_output, self.features)
        print('similarity:', similarity)
        return similarity

class DissimilarityToConceptTarget:
    def __init__(self, features):
        self.features = features
    
    def __call__(self, model_output):
        cos = torch.nn.CosineSimilarity(dim=0)
        dissimilarity = 1-cos(model_output, self.features)
        print('dissimilarity:', dissimilarity)
        return dissimilarity
    
target_layers = [resnet.layer4[-1]]
ref_img_similaritiy = [SimilarityToConceptTarget(ref_img_features)]
target_img_similarity = [SimilarityToConceptTarget(target_img_features)]

# Where is the target image in the reference image
with GradCAM(model=model,
             target_layers=target_layers,
             use_cuda=False) as cam:
    img_target_grayscale_cam = cam(input_tensor=target_img_tensor,
                        targets=ref_img_similaritiy)[0, :]
img_target_cam = show_cam_on_image(target_img_float, img_target_grayscale_cam, use_rgb=True)
path_img_target_similarities = "0_img_target_similarities.png"
print(f'Saving \'{path_img_target_similarities}\'')
Image.fromarray(img_target_cam).save(path_img_target_similarities)

# Where is the reference image in the target image
with GradCAM(model=model,
             target_layers=target_layers,
             use_cuda=False) as cam:
    ref_img_grayscale_cam = cam(input_tensor=ref_tensor,
                        targets=target_img_similarity)[0, :]
img_reference_cam = show_cam_on_image(ref_img_float, ref_img_grayscale_cam, use_rgb=True)
path_img_reference_similarities = "0_img_reference_similarities.png"
print(f'Saving \'{path_img_reference_similarities}\'')
Image.fromarray(img_reference_cam).save(path_img_reference_similarities)


#############################

ref_img_dissimilaritiy = [DissimilarityToConceptTarget(ref_img_features)]
target_img_dissimilarity = [DissimilarityToConceptTarget(target_img_features)]

# Where not the target image is in the reference image
with GradCAM(model=model,
             target_layers=target_layers,
             use_cuda=False) as cam:
    img_target_grayscale_cam = cam(input_tensor=target_img_tensor,
                        targets=ref_img_dissimilaritiy)[0, :]
img_target_cam = show_cam_on_image(target_img_float, img_target_grayscale_cam, use_rgb=True)
path_img_target_dissimilarities = "1_img_target_dissimilarities.png"
print(f'Saving \'{path_img_target_dissimilarities}\'')
Image.fromarray(img_target_cam).save(path_img_target_dissimilarities)

# Where not the reference image is in the target image
with GradCAM(model=model,
             target_layers=target_layers,
             use_cuda=False) as cam:
    ref_img_grayscale_cam = cam(input_tensor=ref_tensor,
                        targets=target_img_dissimilarity)[0, :]
img_reference_cam = show_cam_on_image(ref_img_float, ref_img_grayscale_cam, use_rgb=True)
path_img_reference_dissimilarities = "1_img_reference_dissimilarities.png"
print(f'Saving \'{path_img_reference_dissimilarities}\'')
Image.fromarray(img_reference_cam).save(path_img_reference_dissimilarities)
