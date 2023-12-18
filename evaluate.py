import torch
import torch.nn as nn
from torchvision import models, transforms
import cv2
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from scipy.stats import pearsonr
from PIL import Image
import numpy as np
import os

# Define a simple VGG-based perceptual loss
class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        self.vgg = models.vgg19(pretrained=True).features[:18].eval()
        for param in self.vgg.parameters():
            param.requires_grad = False

    def forward(self, x, y):
        features_x = self.vgg(x)
        features_y = self.vgg(y)
        loss = nn.functional.mse_loss(features_x, features_y)
        return loss
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

val_dir = r'./datasets/cross_sub/val'
test_dir = r'./test'
participant = 'sub-07'

original_path = val_dir + '/' + f'{participant}' +'/C/'
generated_path = test_dir + '/' + f'{participant}' +'/B/'
original_file = os.listdir(original_path)
generated_file = os.listdir(generated_path)

ssim_l = []
psnr_l = []
pear_l = []
perceptual_l = []
perceptual_loss = PerceptualLoss()
for i, (original_name, generated_name) in enumerate(zip(original_file, generated_file)):
    # Read images
    original_image = cv2.imread(original_path + "\\" + original_name)
    generated_image = cv2.imread(generated_path + "\\" + generated_name)

    original_image_gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    generated_image_gray = cv2.cvtColor(generated_image, cv2.COLOR_BGR2GRAY)

    # Calculate SSIM
    ssim_index = ssim(original_image_gray, generated_image_gray)

    # Calculate PSNR
    psnr_value = psnr(original_image_gray, generated_image_gray)

    # Calculate Pearson correlations
    flattened_ori = original_image_gray.flatten()
    flattened_gen = generated_image_gray.flatten()
    r, p = pearsonr(flattened_ori, flattened_gen)

    # Convert images to tensors
    input_tensor = transforms.ToTensor()(Image.fromarray(original_image)).unsqueeze(0)
    target_tensor = transforms.ToTensor()(Image.fromarray(generated_image)).unsqueeze(0)

    # Calculate perceptual loss
    perceptual_loss_val = perceptual_loss(target_tensor, input_tensor)


    ssim_l.append(ssim_index)
    psnr_l.append(psnr_value)
    pear_l.append(r)
    perceptual_l.append(perceptual_loss_val)

ave_ssim = np.mean(ssim_l)
ave_psnr = np.mean(psnr_l)
ave_pear = np.mean(pear_l)
ave_perceptual_loss = np.mean(perceptual_l)
print(f"{participant}_SSIM Index: {ave_ssim}")
print(f"{participant}_PSNR Value: {ave_psnr}")
print(f"{participant}_Pearson correlation Value: {ave_pear}")
print(f"{participant}_Perceptual_loss Value: {ave_perceptual_loss}")