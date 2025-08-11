import torch
import torch.nn as nn
import torch.nn.functional as F
from .unet_dfc_sa_ablation_branches import LightSelfAttention, AblationUNetBase

# 實驗 2(a) DFC-SA Block (相加融合)
class AdditionFusionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, pool_size=8):
        super(AdditionFusionBlock, self).__init__()
        
        # 局部分支：標準卷積模組（基於原始實現）
        self.conv_branch = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                      stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # 自注意力分支（基於原始實現）
        self.attn_branch = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            LightSelfAttention(out_channels, pool_size=pool_size)
        )
        
        # 殘差連接
        if in_channels != out_channels:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        else:
            self.residual_conv = nn.Identity()
            
        self.res_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        # 局部特徵
        local_feat = self.conv_branch(x)
        # 全局自注意力特徵
        attn_feat = self.attn_branch(x)
        
        # 核心修改：簡單相加融合（替代動態門控）
        fused = local_feat + attn_feat
        
        # 加入殘差連接
        res = self.residual_conv(x)
        out = fused + self.res_scale * res
        
        return out

# 實驗 2(b) DFC-SA Block (串接融合)
class ConcatFusionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, pool_size=8):
        super(ConcatFusionBlock, self).__init__()
        
        # 局部分支：標準卷積模組（基於原始實現）
        self.conv_branch = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                      stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # 自注意力分支（基於原始實現）
        self.attn_branch = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            LightSelfAttention(out_channels, pool_size=pool_size)
        )
        
        # 核心修改：串接融合（替代動態門控）
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # 殘差連接
        if in_channels != out_channels:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        else:
            self.residual_conv = nn.Identity()
            
        self.res_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        # 局部特徵
        local_feat = self.conv_branch(x)
        # 全局自注意力特徵
        attn_feat = self.attn_branch(x)
        
        # 核心修改：串接融合（替代動態門控）
        combined = torch.cat([local_feat, attn_feat], dim=1)
        fused = self.fusion_conv(combined)
        
        # 加入殘差連接
        res = self.residual_conv(x)
        out = fused + self.res_scale * res
        
        return out

# 模型定義 (使用與實驗一相同的 AblationUNetBase)
class UNet_AdditionFusion(AblationUNetBase):
    def __init__(self, in_channels, out_channels, features, pool_size=8):
        super().__init__(lambda i, o: AdditionFusionBlock(i, o, pool_size=pool_size), in_channels, out_channels, features, pool_size)

class UNet_ConcatFusion(AblationUNetBase):
    def __init__(self, in_channels, out_channels, features, pool_size=8):
        super().__init__(lambda i, o: ConcatFusionBlock(i, o, pool_size=pool_size), in_channels, out_channels, features, pool_size) 