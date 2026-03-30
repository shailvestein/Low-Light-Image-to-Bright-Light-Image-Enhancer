import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF



class FeatureSelfTransform(nn.Module):
    def __init__(self, channels):
        super(FeatureSelfTransform, self).__init__()
        self.vector_scale = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.vector_bias = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x):
        return x * self.vector_scale + self.vector_bias

class AttentionModule(nn.Module):
    def __init__(self, channels):
        super(AttentionModule, self).__init__()
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
        avg_pool = F.avg_pool2d(x, kernel_size=2, stride=2, padding=0, ceil_mode=False) # Simplistic pooling for pooling box logic
        avg_pool = F.interpolate(avg_pool, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        path_a = self.sigmoid(self.repconv_1x1_a(avg_pool))

        max_pool = F.max_pool2d(x, kernel_size=2, stride=2, padding=0, ceil_mode=False)
        max_pool = F.interpolate(max_pool, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        path_b = self.sigmoid(self.repconv_1x1_b(max_pool))

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
            self.main_branch = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False),
                nn.BatchNorm2d(out_channels)
            )

            branches = []
            if kernel_size == 5:
                extra_kernels = [(3, 3), (1, 1), (3, 1), (1, 3)]
            elif kernel_size == 3:
                extra_kernels = [(1, 1), (3, 1), (1, 3)]
            else:
                extra_kernels = []

            for k in extra_kernels:
                if isinstance(k, tuple):
                    p_h, p_w = k[0]//2, k[1]//2
                    branch = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, k, padding=(p_h, p_w), bias=False),
                        nn.BatchNorm2d(out_channels)
                    )
                else:
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
        output = self.sigmoid(self.final_conv(x6))
        return torch.clamp(output, 0, 1)
