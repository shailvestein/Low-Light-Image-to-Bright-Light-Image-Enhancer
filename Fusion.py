import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF




class Fusion(nn.Module):
    def __init__(self, model_zero, model_ret, model_gmfn):
        super(Fusion, self).__init__()
        self.zero, self.ret, self.gmfn = model_zero, model_ret, model_gmfn
        
        for p in self.zero.parameters(): p.requires_grad = False
        for p in self.ret.parameters(): p.requires_grad = False
        for p in self.gmfn.parameters(): p.requires_grad = False

        self.attention_head = nn.Sequential(
            nn.Conv2d(9, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 3, kernel_size=1),
            nn.Softmax(dim=1) 
        )
        
        self.refine_block = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 3, kernel_size=3, padding=1)
        )

    def forward(self, x):
        with torch.no_grad():
            self.zero.eval(); self.ret.eval(); self.gmfn.eval()
            out_z, _ = self.zero(x) 
            out_r, _ = self.ret(x)  
            out_g = self.gmfn(x)    

        feat_cat = torch.cat([out_z, out_r, out_g], dim=1)
        att_weights = self.attention_head(feat_cat)
        
        fused = (att_weights[:, 0:1] * out_z) + \
                (att_weights[:, 1:2] * out_r) + \
                (att_weights[:, 2:3] * out_g)
        
        refined_output = fused + self.refine_block(fused)
        
        final = (refined_output - 0.05) * 1.12
        return torch.clamp(final, 0, 1)
