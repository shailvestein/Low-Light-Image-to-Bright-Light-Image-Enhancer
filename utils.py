import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torchvision.models import vgg16, VGG16_Weights

import random
import kornia

import os
import time
import sys

import numpy as np
from piq import ssim, psnr
import lpips

import cv2
import gdown
import streamlit as st

batch_size=8
patch_size=256
random_samples = 16

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")


# Double Convolution Block (The building block)
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.PReLU()
        )

    def forward(self, x):
        return self.double_conv(x)

#----------------------------------------------------------------------------------#
#                    R E T I N E X    N E T    M O D E L                           #
#----------------------------------------------------------------------------------#



class RetinexUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(RetinexUNet, self).__init__()
        features = [32, 64, 128, 256]

        # Encoder
        self.inc = DoubleConv(in_channels, features[0])
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(features[0], features[1]))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(features[1], features[2]))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(features[2], features[3]))

        # Bottleneck
        self.bottleneck = nn.Sequential(nn.MaxPool2d(2), DoubleConv(features[3], features[3] * 2))

        # Decoder
        self.up1 = nn.ConvTranspose2d(features[3] * 2, features[3], kernel_size=2, stride=2)
        self.dec1 = DoubleConv(features[3] * 2, features[3])

        self.up2 = nn.ConvTranspose2d(features[3], features[2], kernel_size=2, stride=2)
        self.dec2 = DoubleConv(features[2] * 2, features[2])

        self.up3 = nn.ConvTranspose2d(features[2], features[1], kernel_size=2, stride=2)
        self.dec3 = DoubleConv(features[1] * 2, features[1])

        self.up4 = nn.ConvTranspose2d(features[1], features[0], kernel_size=2, stride=2)
        self.dec4 = DoubleConv(features[0] * 2, features[0])

        # Final Layer to predict Illumination Map
        self.outc = nn.Sequential(
            nn.Conv2d(features[0], out_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x_low):
        # Encoder Pathway
        e1 = self.inc(x_low)
        e2 = self.down1(e1)
        e3 = self.down2(e2)
        e4 = self.down3(e3)
        b = self.bottleneck(e4)

        # Decoder Pathway with Skip Connections
        d1 = self.dec1(torch.cat([e4, self.up1(b)], dim=1))
        d2 = self.dec2(torch.cat([e3, self.up2(d1)], dim=1))
        d3 = self.dec3(torch.cat([e2, self.up3(d2)], dim=1))
        d4 = self.dec4(torch.cat([e1, self.up4(d3)], dim=1))

        # PREDICT ILLUMINATION MAP
        illumination = self.outc(d4)
        illumination = torch.clamp(illumination, min=0.01) # Avoid division by zero

        # APPLY RETINEX: Enhanced = Input / Illumination
        # This keeps the original edges SHARP
        enhanced = x_low / illumination
        enhanced = torch.clamp(enhanced, 0, 1)
        return enhanced, illumination


class UNetTrainer:
    def __init__(self, model, weights_name):
        self.model = model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.best_model_path = weights_name


    def predict(self, x):
        self.model.eval()
        with torch.no_grad():
            return self.model(x.to(self.device))

    def load_best_model(self):
        if not self.best_model_path:
            raise ValueError("No best model found. Please train the model first.")
        self.model.load_state_dict(torch.load(self.best_model_path))


#----------------------------------------------------------------------------------#
#                    Z E R O D C E    N E T    M O D E L                           #
#----------------------------------------------------------------------------------#


class ZeroDCENet(nn.Module):
    def __init__(self, n_iter=8):
        super(ZeroDCENet, self).__init__()
        self.n_iter = n_iter

        # 7-layer CNN with symmetrical skip connections
        # Input: 3 channels (RGB)
        # Output: 3 * n_iter channels (8 iterations * 3 RGB channels = 24)

        self.relu = nn.ReLU(inplace=True)

        self.e_conv1 = nn.Conv2d(3, 32, 3, 1, 1, bias=True)
        self.e_conv2 = nn.Conv2d(32, 32, 3, 1, 1, bias=True)
        self.e_conv3 = nn.Conv2d(32, 32, 3, 1, 1, bias=True)
        self.e_conv4 = nn.Conv2d(32, 32, 3, 1, 1, bias=True)
        self.e_conv5 = nn.Sequential(nn.Conv2d(64, 32, 3, 1, 1, bias=True), self.relu)
        self.e_conv6 = nn.Sequential(nn.Conv2d(64, 32, 3, 1, 1, bias=True), self.relu)
        self.e_conv7 = nn.Sequential(nn.Conv2d(64, 3 * n_iter, 3, 1, 1, bias=True), nn.Tanh())

    def forward(self, x):
        # Encoder with Skip Connections
        x1 = self.relu(self.e_conv1(x))
        x2 = self.relu(self.e_conv2(x1))
        x3 = self.relu(self.e_conv3(x2))
        x4 = self.relu(self.e_conv4(x3))

        # Decoder
        x5 = self.e_conv5(torch.cat([x3, x4], 1))
        x6 = self.e_conv6(torch.cat([x2, x5], 1))
        alpha_map = self.e_conv7(torch.cat([x1, x6], 1)) # [B, 24, H, W]

        # Iterative Enhancement
        y = x
        # Split the 24 channels into 8 sets of 3 RGB channels
        for i in range(self.n_iter):
            a = alpha_map[:, i*3 : (i+1)*3, :, :]
            y = y + a * y * (1 - y)
        return y, alpha_map


class DCETrainer:
    def __init__(self, model, weights_name):
        self.model = model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.best_model_path = weights_name

    def predict(self, x):
        with torch.no_grad():
            return self.model(x.to(self.device))


    def load_best_model(self):
        if not self.best_model_path:
            raise ValueError("No best model found. Please train the model first.")
        self.model.load_state_dict(torch.load(self.best_model_path))



#----------------------------------------------------------------------------------#
#                    F U S I O N    N E T    M O D E L                             #
#----------------------------------------------------------------------------------#

class DenoiseBlock(nn.Module):
    def __init__(self, channels):
        super(DenoiseBlock, self).__init__()
        # Use Dilated Convolutions to see more context without losing resolution
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=2, dilation=2),
            nn.PReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.PReLU()
        )
    def forward(self, x):
        return x + self.conv(x) # Residual connection preserves sharpness

class ColorCorrection(nn.Module):
    def __init__(self):
        super(ColorCorrection, self).__init__()
        self.pointwise = nn.Sequential(
            nn.Conv2d(3, 16, 1),
            nn.PReLU(),
            nn.Conv2d(16, 3, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        # To subtle scale image colors
        return x * self.pointwise(x)

class FusionNet(nn.Module):
    def __init__(self, retinex_model, zerodce_model):
        super(FusionNet, self).__init__()

        self.retinex_net = retinex_model
        self.zerodce_net = zerodce_model

        # Pre-trained models freeze for eval mode
        for param in self.retinex_net.parameters():
            param.requires_grad = False
        for param in self.zerodce_net.parameters():
            param.requires_grad = False

        self.retinex_net.eval()
        self.zerodce_net.eval()

        # Channels: Retinex(3) + ZeroDCE(3) + Alpha(24) + Input(3) = 33
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(33, 64, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.PReLU()
        )

        # Attention mask it decides which model works better where
        self.attention_mask = nn.Sequential(
            nn.Conv2d(32, 3, kernel_size=1),
            nn.Sigmoid()
        )

        # 2. Refinement head to remove grayish look and fix contrast level
        self.refiner = nn.Sequential(
            nn.Conv2d(32, 3, kernel_size=1),
            nn.Tanh() # between -1 to 1 for "fine-tuning"
        )


    def forward(self, x):
        with torch.no_grad():
            out_retinex, _ = self.retinex_net(x)
            out_zerodce, alpha_map = self.zerodce_net(x)

        # Fuse every features from retinex and zero dce models
        combined_features = torch.cat([out_retinex, out_zerodce, alpha_map, x], dim=1)
        feat = self.feature_extractor(combined_features)

        # Attention Blending to preserve sharpness
        # mask = 1 means Retinex heavy, and mask = 0 means ZeroDCE heavy
        mask = self.attention_mask(feat)
        fused = mask * out_retinex + (1 - mask) * out_zerodce

        # Residual Refinement (to remoce greyish tone)
        # Adding a little learned contrast in fusion
        refinement = self.refiner(feat)
        final_output = fused + 0.1 * refinement # 0.1 for factor stability

        return torch.clamp(final_output, 0, 1), alpha_map


class FusedTrainer:
    def __init__(self, model, weights_name):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.best_model_path = weights_name
        self.denoiser = DenoiseBlock(3).to(self.device)
        self.color_corrector = ColorCorrection().to(self.device)

    def predict(self, x):
        with torch.no_grad():
            return self.model(x.to(self.device))


    def load_best_model(self):
        if not self.best_model_path:
            raise ValueError("No best model found. Please train the model first.")
        self.model.load_state_dict(torch.load(self.best_model_path))


#----------------------------------------------------------------------------------#
#            D O W N L O A D    P R E - T R A I N E D   W E I G H T S              #
#----------------------------------------------------------------------------------#

dcenet_model_id = "1P4lhymUpgj2Zc466kz-9815MajpbOgZL"
fused_model_id = "1LEGeO9NuFckR3I8JMidICaVZY8ByTHEj"
unet_model_id = "1WQUO4XYAjHNEjNlnPhXlKhS7wHlkumK2"

unet_model_name = "best-unet-model.pth"
dcenet_model_name = "best-dcenet-model.pth"
fused_model_name = "best-fused-model.pth"

def download_weights(file_id, model_name):
    url = f'https://drive.google.com/uc?id={file_id}'
    if not os.path.exists(model_name):
        with st.spinner(f"Downloading model {model_name} weights from Google Drive..."):
            gdown.download(url, model_name, quiet=False)
    return model_name

def load_weights():
    download_weights(unet_model_id, unet_model_name)
    unet = UNetTrainer(RetinexNet(), unet_model_name)

    download_weights(dcenet_model_id, dcenet_model_name)
    dcenet = DCENetTrainer(ZeroDCENet(n_iter=8), dcenet_mode_name)

    download_weights(fused_model_id, fused_model_name)
    fusednet = FusedTrainer(FusionNet(unet, dcenet), fused_model_name)

    return fusednet

