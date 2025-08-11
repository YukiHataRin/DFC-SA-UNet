import torch
import torch.nn as nn
import torch.nn.functional as F
from .unet_dfc_sa_ablation_branches import AblationUNetBase

# 實驗 3: 完整解析度的自注意力
class FullResolutionAttention(nn.Module):
    def __init__(self, channels, **kwargs):  # 忽略 pool_size
        super(FullResolutionAttention, self).__init__()
        self.query_conv = nn.Conv2d(channels, channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(channels, channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(channels, channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
    
    def forward(self, x):
        B, C, H, W = x.shape
        # 核心修改：直接在原解析度上計算注意力
        proj_query = self.query_conv(x).view(B, -1, H * W).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(B, -1, H * W)
        energy = torch.bmm(proj_query, proj_key)
        attention = F.softmax(energy, dim=-1)
        proj_value = self.value_conv(x).view(B, -1, H * W)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(B, C, H, W)
        out = self.gamma * out + x
        return out

# 使用完整解析度自注意力的 DFC-SA Block（基於原始實現）
class FullResAttnDFCBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, **kwargs):
        super(FullResAttnDFCBlock, self).__init__()
        
        # 局部分支：標準卷積模組（基於原始實現）
        self.conv_branch = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                      stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # 自注意力分支：使用完整解析度的自注意力（核心修改）
        self.attn_branch = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            FullResolutionAttention(out_channels)
        )
        
        # 動態門控：根據兩分支輸出自動學習融合權重（保持原始實現）
        self.gate = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.Sigmoid()
        )
        
        # 融合後再做一次投射（保持原始實現）
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(out_channels * 3, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # 殘差連接（保持原始實現）
        if in_channels != out_channels:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        else:
            self.residual_conv = nn.Identity()
        
        self.res_scale = nn.Parameter(torch.tensor(0.1))
    
    def forward(self, x):
        # 局部特徵
        local_feat = self.conv_branch(x)
        # 全局自注意力特徵（完整解析度 - 核心修改）
        attn_feat = self.attn_branch(x)
        
        # 將兩者拼接（保持原始實現）
        combined = torch.cat([local_feat, attn_feat], dim=1)
        # 計算融合門控權重（保持原始實現）
        gate_weight = self.gate(combined)
        # 利用門控權重動態融合（保持原始實現）
        fused = gate_weight * local_feat + (1 - gate_weight) * attn_feat
        
        # 將融合結果與原始拼接的資訊做線性變換（保持原始實現）
        fusion_input = torch.cat([fused, combined], dim=1)
        out = self.fusion_conv(fusion_input)
        
        # 加入殘差連接（保持原始實現）
        res = self.residual_conv(x)
        out = out + self.res_scale * res
        
        return out

# 模型定義
class UNet_FullResAttention(AblationUNetBase):
    def __init__(self, in_channels, out_channels, features, **kwargs):
        super().__init__(lambda i, o: FullResAttnDFCBlock(i, o), in_channels, out_channels, features, **kwargs) 