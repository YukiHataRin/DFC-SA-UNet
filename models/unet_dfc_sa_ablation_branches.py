import torch
import torch.nn as nn
import torch.nn.functional as F

class LightSelfAttention(nn.Module):
    def __init__(self, channels, pool_size=8):
        """
        輕量化自注意力模組
        參數:
            channels: 輸入通道數
            pool_size: 下採樣後的解析度
        """
        super(LightSelfAttention, self).__init__()
        self.pool_size = pool_size
        self.query_conv = nn.Conv2d(channels, channels // 8, kernel_size=1)
        self.key_conv   = nn.Conv2d(channels, channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(channels, channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
    
    def forward(self, x):
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        # 下採樣至固定解析度
        pooled = F.adaptive_avg_pool2d(x, (self.pool_size, self.pool_size))
        B, C, H_p, W_p = pooled.shape
        
        # 計算 query, key 與 value（在下採樣後的特徵上）
        proj_query = self.query_conv(pooled).view(B, -1, H_p * W_p).permute(0, 2, 1)  # [B, N, C']
        proj_key   = self.key_conv(pooled).view(B, -1, H_p * W_p)                   # [B, C', N]
        energy = torch.bmm(proj_query, proj_key)  # [B, N, N]
        attention = F.softmax(energy, dim=-1)
        proj_value = self.value_conv(pooled).view(B, C, H_p * W_p)  # [B, C, N]
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))  # [B, C, N]
        out = out.view(B, C, H_p, W_p)
        # 上採樣回原尺寸
        out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)
        # 殘差連接與學習縮放參數
        out = self.gamma * out + x
        return out

# 實驗 1(a) DFC-SA Block (僅注意力分支)
class AttentionOnlyBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, pool_size=8):
        super(AttentionOnlyBlock, self).__init__()
        
        # 自注意力分支：基於 unet_dfc_sa_res.py 的實現
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
        # 僅使用自注意力分支
        attn_feat = self.attn_branch(x)
        
        # 加入殘差連接
        res = self.residual_conv(x)
        out = attn_feat + self.res_scale * res
        
        return out

# 實驗 1(b) DFC-SA Block (僅局部分支) / 標準 UNet 卷積塊
class LocalOnlyBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, **kwargs):
        super(LocalOnlyBlock, self).__init__()
        
        # 局部分支：基於 unet_dfc_sa_res.py 的實現
        self.conv_branch = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                      stride=stride, padding=padding),
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
        # 僅使用局部分支
        local_feat = self.conv_branch(x)
        
        # 加入殘差連接
        res = self.residual_conv(x)
        out = local_feat + self.res_scale * res
        
        return out

# 通用 U-Net 框架
class AblationUNetBase(nn.Module):
    def __init__(self, block_func, in_channels, out_channels, features, pool_size=8):
        super(AblationUNetBase, self).__init__()
        self.down1 = block_func(in_channels, features[0])
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = block_func(features[0], features[1])
        self.pool2 = nn.MaxPool2d(2)
        self.down3 = block_func(features[1], features[2])
        self.pool3 = nn.MaxPool2d(2)
        self.down4 = block_func(features[2], features[3])
        self.pool4 = nn.MaxPool2d(2)
        
        self.bottleneck = block_func(features[3], features[3] * 2)

        self.up4 = nn.ConvTranspose2d(features[3] * 2, features[3], kernel_size=2, stride=2)
        self.up_conv4 = block_func(features[3] * 2, features[3])
        self.up3 = nn.ConvTranspose2d(features[3], features[2], kernel_size=2, stride=2)
        self.up_conv3 = block_func(features[2] * 2, features[2])
        self.up2 = nn.ConvTranspose2d(features[2], features[1], kernel_size=2, stride=2)
        self.up_conv2 = block_func(features[1] * 2, features[1])
        self.up1 = nn.ConvTranspose2d(features[1], features[0], kernel_size=2, stride=2)
        self.up_conv1 = block_func(features[0] * 2, features[0])
        
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        d1 = self.down1(x)
        p1 = self.pool1(d1)
        d2 = self.down2(p1)
        p2 = self.pool2(d2)
        d3 = self.down3(p2)
        p3 = self.pool3(d3)
        d4 = self.down4(p3)
        p4 = self.pool4(d4)
        bn = self.bottleneck(p4)
        
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

class UNet_Baseline(AblationUNetBase):
    def __init__(self, in_channels, out_channels, features, **kwargs):
        super().__init__(lambda i, o: LocalOnlyBlock(i, o), in_channels, out_channels, features)

class UNet_AttentionOnly(AblationUNetBase):
    def __init__(self, in_channels, out_channels, features, pool_size=8):
        super().__init__(lambda i, o: AttentionOnlyBlock(i, o, pool_size=pool_size), in_channels, out_channels, features, pool_size) 