
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from einops import rearrange


class LayerNorm(nn.Module):
    def __init__(self, channels):
        super(LayerNorm, self).__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        return x.permute(0, 3, 1, 2)

class MDTA(nn.Module):
    def __init__(self, dim, num_heads=8, bias=False):
        super(MDTA, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, illum):
        b, c, h, w = x.shape
        x = x * illum 
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', h=h, w=w)
        return self.project_out(out)

class GFF(nn.Module):
    def __init__(self, dim, expansion_factor=2.0, bias=False):
        super(GFF, self).__init__()
        hidden_features = int(dim * expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1, groups=hidden_features * 2, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        return self.project_out(x)

class Retinexformer(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, n_feat=64):
        super(Retinexformer, self).__init__()
        
        self.estimator = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 1, 3, 1, 1),
            nn.Sigmoid()
        )

        self.embed = nn.Conv2d(in_channels, n_feat, 3, 1, 1)

        self.denoiser = nn.Sequential(
            nn.Conv2d(n_feat, n_feat, kernel_size=3, padding=1, groups=n_feat), # Depthwise
            nn.GELU(),
            nn.Conv2d(n_feat, n_feat, kernel_size=1)
        )
        
        self.norm1 = LayerNorm(n_feat)
        self.attn1 = MDTA(n_feat)
        self.norm2 = LayerNorm(n_feat)
        self.ffn1 = GFF(n_feat)
        self.output = nn.Conv2d(n_feat, out_channels, 3, 1, 1)

    def forward(self, x):
        illum = self.estimator(x)
        illum_map = torch.clamp(illum, 0.1, 0.9) 
        
        fea = self.embed(x)
        fea = self.denoiser(fea)
        fea = fea + self.attn1(self.norm1(fea), illum_map)
        fea = fea + self.ffn1(self.norm2(fea))
        r_out = torch.sigmoid(self.output(fea))
        enhanced = (x * 0.2) + (r_out * illum_map.repeat(1, 3, 1, 1) * 0.8)
        enhanced = torch.pow(enhanced, 0.8) 
        
        return torch.clamp(enhanced, 0, 1), illum_map
