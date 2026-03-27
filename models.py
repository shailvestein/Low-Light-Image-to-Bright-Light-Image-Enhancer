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

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")


class FeatureSelfTransform(nn.Module):
    def __init__(self, channels):
        super(FeatureSelfTransform, self).__init__()
        self.vector_scale = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.vector_bias = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x):
        # x * vector_scale + vector_bias
        return x * self.vector_scale + self.vector_bias

class AttentionModule(nn.Module):
    def __init__(self, channels):
        super(AttentionModule, self).__init__()
        # 1x1 RepConv (Simplification for training)
        self.repconv_1x1_a = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels)
        )
        self.repconv_1x1_b = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Path A (Average Pooling)
        avg_pool = F.avg_pool2d(x, kernel_size=2, stride=2, padding=0, ceil_mode=False) # Simplistic pooling for pooling box logic
        # Pad back to match shape or use AdaptiveAvgPool. Let's keep it simple for now based on diagram
        avg_pool = F.interpolate(avg_pool, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        path_a = self.sigmoid(self.repconv_1x1_a(avg_pool))

        # Path B (Max Pooling)
        max_pool = F.max_pool2d(x, kernel_size=2, stride=2, padding=0, ceil_mode=False)
        # Pad back
        max_pool = F.interpolate(max_pool, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        path_b = self.sigmoid(self.repconv_1x1_b(max_pool))

        # Multiply Path A & B outputs with the original features as in diagram
        return x * path_a * path_b

class MBRConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, deploy=False):
        super(MBRConvBlock, self).__init__()
        self.deploy = deploy
        self.kernel_size = kernel_size
        padding = kernel_size // 2

        if deploy:
            self.reparam_conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=True)
        else:
            # Common Branch (Central Kernel Size e.g., 5x5 or 3x3)
            self.main_branch = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False),
                nn.BatchNorm2d(out_channels)
            )

            # Extra Branches as per diagram MBRConv detail
            branches = []
            if kernel_size == 5:
                # Branches: 3x3, 1x1, 3x1, 1x3
                extra_kernels = [(3, 3), (1, 1), (3, 1), (1, 3)]
            elif kernel_size == 3:
                # Branches: 1x1, 3x1, 1x3
                extra_kernels = [(1, 1), (3, 1), (1, 3)]
            else: # kernel_size == 1 case for Attention Module but typically handled within there. Let's not make it more complex here.
                extra_kernels = []

            for k in extra_kernels:
                if isinstance(k, tuple): # Asymmetric kernels e.g., 3x1
                    p_h, p_w = k[0]//2, k[1]//2
                    branch = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, k, padding=(p_h, p_w), bias=False),
                        nn.BatchNorm2d(out_channels)
                    )
                else: # Symmetric kernels like 1x1
                    branch = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, k, padding=k//2, bias=False),
                        nn.BatchNorm2d(out_channels)
                    )
                branches.append(branch)
            self.extra_branches = nn.ModuleList(branches)

    def forward(self, x):
        if self.deploy:
            return self.reparam_conv(x)
        else:
            out = self.main_branch(x)
            for branch in self.extra_branches:
                out += branch(x)
            return out


class GeneralMethodFlowNetwork(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, base_channels=64, deploy=False):
        super(GeneralMethodFlowNetwork, self).__init__()
        self.pixel_unshuffle = nn.PixelUnshuffle(2)
        mid_channels = in_channels * (2**2)

        self.repconv_5x5 = MBRConvBlock(mid_channels, base_channels, kernel_size=5, deploy=deploy)
        self.prelu = nn.PReLU()

        self.repconv_3x3_fst1 = MBRConvBlock(base_channels, base_channels, kernel_size=3, deploy=deploy)
        self.fst1 = FeatureSelfTransform(base_channels)

        self.repconv_3x3_fst2 = MBRConvBlock(base_channels, base_channels, kernel_size=3, deploy=deploy)
        self.fst2 = FeatureSelfTransform(base_channels)
        self.attention = AttentionModule(base_channels)

        self.pre_shuffle_reduce = nn.Conv2d(base_channels, 16, kernel_size=1, bias=False)
        self.pixel_shuffle = nn.PixelShuffle(2)

        # --- UPDATE: Final Layer without BatchNorm + Sigmoid ---
        self.final_conv = nn.Conv2d(4, out_channels, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.pixel_unshuffle(x)
        x2 = self.repconv_5x5(x1)
        x3 = self.prelu(x2)
        x4 = self.repconv_3x3_fst1(x3)
        x4_fst = self.fst1(x4)
        x5 = self.repconv_3x3_fst2(x4_fst)
        x5_fst = self.fst2(x5)
        x5_atten = self.attention(x5_fst)
        x6_reduce = self.pre_shuffle_reduce(x5_atten)
        x6 = self.pixel_shuffle(x6_reduce)

        # --- UPDATE: Apply Sigmoid for stability ---
        output = self.sigmoid(self.final_conv(x6))
        return output

    @torch.no_grad()
    def predict(self, x):
        self.eval()
        device = next(self.parameters()).device
        output = self.forward(x.to(device))
        return output # Sigmoid already handles 0-1



#----------------------------------------------------------------------------------#
#                    R E T I N E X    N E T    M O D E L                           #
#----------------------------------------------------------------------------------#


def conv_relu(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
        nn.ReLU(inplace=True)
    )

class RetinexUNet(nn.Module):
    def __init__(self):
        super(RetinexUNet, self).__init__()
        
        # --- Encoder ---
        self.enc1 = conv_relu(3, 64)
        self.enc2 = conv_relu(64, 128)
        self.enc3 = conv_relu(128, 256)
        self.pool = nn.MaxPool2d(2)

        # --- Decoder ---
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = conv_relu(256, 128) # 128 (up) + 128 (skip)
        
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = conv_relu(128, 64) # 64 (up) + 64 (skip)

        # --- Output Heads ---
        # 3 Channels for Reflectance (R)
        self.reflectance_head = nn.Conv2d(64, 3, kernel_size=3, padding=1)
        # 1 Channel for Illumination (L) - Light grayscale hoti hai
        self.illumination_head = nn.Conv2d(64, 1, kernel_size=3, padding=1)

    def forward(self, x):
        # Encoder with skip connections
        s1 = self.enc1(x)
        p1 = self.pool(s1)
        
        s2 = self.enc2(p1)
        p2 = self.pool(s2)
        
        b = self.enc3(p2) # Bridge/Bottleneck

        # Decoder
        d2 = self.up2(b)
        d2 = torch.cat([d2, s2], dim=1) # Skip connection
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)
        d1 = torch.cat([d1, s1], dim=1) # Skip connection
        d1 = self.dec1(d1)

        # Retinex components
        R = torch.sigmoid(self.reflectance_head(d1))
        L = torch.sigmoid(self.illumination_head(d1))
        
        # Illumination ko 3-channel banayein multiplication ke liye
        L_3ch = L.repeat(1, 3, 1, 1)
        
        # Reconstruction: I = R * L
        enhanced = R * L_3ch
        
        return R, L_3ch, enhanced



#----------------------------------------------------------------------------------#
#            D O W N L O A D    P R E - T R A I N E D   W E I G H T S              #
#----------------------------------------------------------------------------------#

retinex_model_id = "1-QqQFKYESZ1vbKL64VUFghEKex16_WCe"
gmfn_model_id = "1wXSlaCnqv8Byj385PHnz7esjBRzyWuFh"

retinex_model_name = "rank-16-model-ret.pth"
gmfn_model_name = "rank-16-model.pth"

def download_weights(file_id, model_name):
    url = f'https://drive.google.com/uc?id={file_id}'
    if not os.path.exists(model_name):
        with st.spinner(f"Downloading model {model_name} weights from Google Drive..."):
            gdown.download(url, model_name, quiet=False)
    return model_name

#----------------------------------------------------------------------------------#
#            L O A D I N G    P R E - T R A I N E D   W E I G H T S                #
#----------------------------------------------------------------------------------#

def load_weights():
    download_weights(gmfn_model_id, gmfn_model_name)
    gmfn_model = GeneralMethodFlowNetwork
    state_dict = torch.load(gmfn_model_name, map_location=device)
    gmfn_model.load_state_dict(state_dict)
    
    download_weights(retinex_model_id, retinex_model_name)
    retinex_model = ZeroDCENet(n_iter=8)
    state_dict = torch.load(retinex_model_name, map_location=device)
    retinex_model.load_state_dict(state_dict)


    return gmfn_model, retinex_model

