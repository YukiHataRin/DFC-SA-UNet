import torch
import torch.nn as nn
import torch.nn.functional as F
from .unet_dfc_sa_ablation_branches import LightSelfAttention, LocalOnlyBlock

# 動態融合卷積自注意力區塊（完整版 - 直接複製自原始實現）
class DynamicFusionConvAttnBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, pool_size=8):
        """
        動態融合卷積自注意力區塊（改進版）
        
        參數:
            in_channels: 輸入通道數
            out_channels: 輸出通道數（兩個分支皆投射到相同維度）
            kernel_size: 卷積核大小
            stride: 卷積步長
            padding: 卷積填充
            pool_size: 輕量自注意力模組下採樣後的解析度
        """
        super(DynamicFusionConvAttnBlock, self).__init__()
        
        # 局部分支：標準卷積模組
        self.conv_branch = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                      stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # 自注意力分支：先透過 1x1 投射到 out_channels，再進行輕量自注意力
        self.attn_branch = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            LightSelfAttention(out_channels, pool_size=pool_size)
        )
        
        # 動態門控：根據兩分支輸出自動學習融合權重
        self.gate = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.Sigmoid()
        )
        
        # 融合後再做一次投射
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(out_channels * 3, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # 殘差連接：若輸入通道與輸出通道不一致，則用 1x1 卷積進行投射；否則直接 Identity
        if in_channels != out_channels:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        else:
            self.residual_conv = nn.Identity()
        
        # 新增殘差縮放參數，初始值較小，讓殘差分支影響逐漸增加
        self.res_scale = nn.Parameter(torch.tensor(0.1))
    
    def forward(self, x):
        # 局部特徵
        local_feat = self.conv_branch(x)   # [B, out_channels, H, W]
        # 全局自注意力特徵
        attn_feat = self.attn_branch(x)      # [B, out_channels, H, W]
        
        # 將兩者拼接
        combined = torch.cat([local_feat, attn_feat], dim=1)  # [B, 2*out_channels, H, W]
        # 計算融合門控權重（輸出值介於 0～1）
        gate_weight = self.gate(combined)  # [B, out_channels, H, W]
        # 利用門控權重動態融合：局部與全局分支各自加權
        fused = gate_weight * local_feat + (1 - gate_weight) * attn_feat
        
        # 將融合結果與原始拼接的資訊做線性變換
        fusion_input = torch.cat([fused, combined], dim=1)  # [B, 3*out_channels, H, W]
        out = self.fusion_conv(fusion_input)
        
        # 加入殘差連接，並透過縮放參數調整其影響
        res = self.residual_conv(x)
        out = out + self.res_scale * res
        
        return out

# 實驗 4(a) 僅編碼器使用 DFC-SA
class UNet_EncoderOnlyDFC(nn.Module):
    def __init__(self, in_channels, out_channels, features, pool_size=8):
        super().__init__()
        # 編碼器使用 DFC-SA
        self.down1 = DynamicFusionConvAttnBlock(in_channels, features[0], pool_size=pool_size)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = DynamicFusionConvAttnBlock(features[0], features[1], pool_size=pool_size)
        self.pool2 = nn.MaxPool2d(2)
        self.down3 = DynamicFusionConvAttnBlock(features[1], features[2], pool_size=pool_size)
        self.pool3 = nn.MaxPool2d(2)
        self.down4 = DynamicFusionConvAttnBlock(features[2], features[3], pool_size=pool_size)
        self.pool4 = nn.MaxPool2d(2)
        
        self.bottleneck = DynamicFusionConvAttnBlock(features[3], features[3] * 2, pool_size=pool_size)

        # 解碼器使用標準卷積
        self.up4 = nn.ConvTranspose2d(features[3] * 2, features[3], kernel_size=2, stride=2)
        self.up_conv4 = LocalOnlyBlock(features[3] * 2, features[3])
        self.up3 = nn.ConvTranspose2d(features[3], features[2], kernel_size=2, stride=2)
        self.up_conv3 = LocalOnlyBlock(features[2] * 2, features[2])
        self.up2 = nn.ConvTranspose2d(features[2], features[1], kernel_size=2, stride=2)
        self.up_conv2 = LocalOnlyBlock(features[1] * 2, features[1])
        self.up1 = nn.ConvTranspose2d(features[1], features[0], kernel_size=2, stride=2)
        self.up_conv1 = LocalOnlyBlock(features[0] * 2, features[0])
        
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        # 編碼器
        d1 = self.down1(x)
        p1 = self.pool1(d1)
        d2 = self.down2(p1)
        p2 = self.pool2(d2)
        d3 = self.down3(p2)
        p3 = self.pool3(d3)
        d4 = self.down4(p3)
        p4 = self.pool4(d4)
        bn = self.bottleneck(p4)
        
        # 解碼器
        u4 = self.up4(bn)
        if u4.shape[2:] != d4.shape[2:]:
            u4 = F.interpolate(u4, size=d4.shape[2:], mode='bilinear', align_corners=False)
        u4 = torch.cat([u4, d4], dim=1)
        u4 = self.up_conv4(u4)
        
        u3 = self.up3(u4)
        if u3.shape[2:] != d3.shape[2:]:
            u3 = F.interpolate(u3, size=d3.shape[2:], mode='bilinear', align_corners=False)
        u3 = torch.cat([u3, d3], dim=1)
        u3 = self.up_conv3(u3)
        
        u2 = self.up2(u3)
        if u2.shape[2:] != d2.shape[2:]:
            u2 = F.interpolate(u2, size=d2.shape[2:], mode='bilinear', align_corners=False)
        u2 = torch.cat([u2, d2], dim=1)
        u2 = self.up_conv2(u2)
        
        u1 = self.up1(u2)
        if u1.shape[2:] != d1.shape[2:]:
            u1 = F.interpolate(u1, size=d1.shape[2:], mode='bilinear', align_corners=False)
        u1 = torch.cat([u1, d1], dim=1)
        u1 = self.up_conv1(u1)
        
        return self.final_conv(u1)

# 實驗 4(b) 僅解碼器使用 DFC-SA
class UNet_DecoderOnlyDFC(nn.Module):
    def __init__(self, in_channels, out_channels, features, pool_size=8):
        super().__init__()
        # 編碼器使用標準卷積
        self.down1 = LocalOnlyBlock(in_channels, features[0])
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = LocalOnlyBlock(features[0], features[1])
        self.pool2 = nn.MaxPool2d(2)
        self.down3 = LocalOnlyBlock(features[1], features[2])
        self.pool3 = nn.MaxPool2d(2)
        self.down4 = LocalOnlyBlock(features[2], features[3])
        self.pool4 = nn.MaxPool2d(2)
        
        self.bottleneck = LocalOnlyBlock(features[3], features[3] * 2)

        # 解碼器使用 DFC-SA
        self.up4 = nn.ConvTranspose2d(features[3] * 2, features[3], kernel_size=2, stride=2)
        self.up_conv4 = DynamicFusionConvAttnBlock(features[3] * 2, features[3], pool_size=pool_size)
        self.up3 = nn.ConvTranspose2d(features[3], features[2], kernel_size=2, stride=2)
        self.up_conv3 = DynamicFusionConvAttnBlock(features[2] * 2, features[2], pool_size=pool_size)
        self.up2 = nn.ConvTranspose2d(features[2], features[1], kernel_size=2, stride=2)
        self.up_conv2 = DynamicFusionConvAttnBlock(features[1] * 2, features[1], pool_size=pool_size)
        self.up1 = nn.ConvTranspose2d(features[1], features[0], kernel_size=2, stride=2)
        self.up_conv1 = DynamicFusionConvAttnBlock(features[0] * 2, features[0], pool_size=pool_size)
        
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
        
    def forward(self, x):
        # 編碼器
        d1 = self.down1(x)
        p1 = self.pool1(d1)
        d2 = self.down2(p1)
        p2 = self.pool2(d2)
        d3 = self.down3(p2)
        p3 = self.pool3(d3)
        d4 = self.down4(p3)
        p4 = self.pool4(d4)
        bn = self.bottleneck(p4)
        
        # 解碼器
        u4 = self.up4(bn)
        if u4.shape[2:] != d4.shape[2:]:
            u4 = F.interpolate(u4, size=d4.shape[2:], mode='bilinear', align_corners=False)
        u4 = torch.cat([u4, d4], dim=1)
        u4 = self.up_conv4(u4)
        
        u3 = self.up3(u4)
        if u3.shape[2:] != d3.shape[2:]:
            u3 = F.interpolate(u3, size=d3.shape[2:], mode='bilinear', align_corners=False)
        u3 = torch.cat([u3, d3], dim=1)
        u3 = self.up_conv3(u3)
        
        u2 = self.up2(u3)
        if u2.shape[2:] != d2.shape[2:]:
            u2 = F.interpolate(u2, size=d2.shape[2:], mode='bilinear', align_corners=False)
        u2 = torch.cat([u2, d2], dim=1)
        u2 = self.up_conv2(u2)
        
        u1 = self.up1(u2)
        if u1.shape[2:] != d1.shape[2:]:
            u1 = F.interpolate(u1, size=d1.shape[2:], mode='bilinear', align_corners=False)
        u1 = torch.cat([u1, d1], dim=1)
        u1 = self.up_conv1(u1)
        
        return self.final_conv(u1)

# 實驗 4(c) 編碼器和解碼器都使用標準卷積
class UNet_BothStandardConv(nn.Module):
    def __init__(self, in_channels, out_channels, features, **kwargs):
        super().__init__()
        # 編碼器使用標準卷積
        self.down1 = LocalOnlyBlock(in_channels, features[0])
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = LocalOnlyBlock(features[0], features[1])
        self.pool2 = nn.MaxPool2d(2)
        self.down3 = LocalOnlyBlock(features[1], features[2])
        self.pool3 = nn.MaxPool2d(2)
        self.down4 = LocalOnlyBlock(features[2], features[3])
        self.pool4 = nn.MaxPool2d(2)
        
        self.bottleneck = LocalOnlyBlock(features[3], features[3] * 2)

        # 解碼器也使用標準卷積
        self.up4 = nn.ConvTranspose2d(features[3] * 2, features[3], kernel_size=2, stride=2)
        self.up_conv4 = LocalOnlyBlock(features[3] * 2, features[3])
        self.up3 = nn.ConvTranspose2d(features[3], features[2], kernel_size=2, stride=2)
        self.up_conv3 = LocalOnlyBlock(features[2] * 2, features[2])
        self.up2 = nn.ConvTranspose2d(features[2], features[1], kernel_size=2, stride=2)
        self.up_conv2 = LocalOnlyBlock(features[1] * 2, features[1])
        self.up1 = nn.ConvTranspose2d(features[1], features[0], kernel_size=2, stride=2)
        self.up_conv1 = LocalOnlyBlock(features[0] * 2, features[0])
        
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
        
    def forward(self, x):
        # 編碼器
        d1 = self.down1(x)
        p1 = self.pool1(d1)
        d2 = self.down2(p1)
        p2 = self.pool2(d2)
        d3 = self.down3(p2)
        p3 = self.pool3(d3)
        d4 = self.down4(p3)
        p4 = self.pool4(d4)
        bn = self.bottleneck(p4)
        
        # 解碼器
        u4 = self.up4(bn)
        if u4.shape[2:] != d4.shape[2:]:
            u4 = F.interpolate(u4, size=d4.shape[2:], mode='bilinear', align_corners=False)
        u4 = torch.cat([u4, d4], dim=1)
        u4 = self.up_conv4(u4)
        
        u3 = self.up3(u4)
        if u3.shape[2:] != d3.shape[2:]:
            u3 = F.interpolate(u3, size=d3.shape[2:], mode='bilinear', align_corners=False)
        u3 = torch.cat([u3, d3], dim=1)
        u3 = self.up_conv3(u3)
        
        u2 = self.up2(u3)
        if u2.shape[2:] != d2.shape[2:]:
            u2 = F.interpolate(u2, size=d2.shape[2:], mode='bilinear', align_corners=False)
        u2 = torch.cat([u2, d2], dim=1)
        u2 = self.up_conv2(u2)
        
        u1 = self.up1(u2)
        if u1.shape[2:] != d1.shape[2:]:
            u1 = F.interpolate(u1, size=d1.shape[2:], mode='bilinear', align_corners=False)
        u1 = torch.cat([u1, d1], dim=1)
        u1 = self.up_conv1(u1)
        
        return self.final_conv(u1)