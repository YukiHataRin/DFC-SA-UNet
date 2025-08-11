import torch
import torch.nn as nn
import torch.nn.functional as F

class LightSelfAttention(nn.Module):
    def __init__(self, channels, pool_size=8, ablation_on_qk_channels = 8):
        """
        輕量化自注意力模組
        參數:
            channels: 輸入通道數
            pool_size: 下採樣後的解析度
        """
        super(LightSelfAttention, self).__init__()
        self.pool_size = pool_size
        self.query_conv = nn.Conv2d(channels, channels // ablation_on_qk_channels, kernel_size=1)
        self.key_conv   = nn.Conv2d(channels, channels // ablation_on_qk_channels, kernel_size=1)
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

class DynamicFusionConvAttnBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, pool_size=8, ablation_on_qk_channels=8):
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
            LightSelfAttention(out_channels, pool_size=pool_size, ablation_on_qk_channels=ablation_on_qk_channels)
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

class UNetDFCSA(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512], pool_size=8, ablation_on_qk_channels=8):
        """
        整合 DynamicFusionConvAttnBlock 的 U-Net 模型
        
        參數:
            in_channels: 輸入通道數
            out_channels: 輸出通道數
            features: 各層特徵通道數列表
            pool_size: 輕量自注意力模組下採樣後的解析度
        """
        super(UNetDFCSA, self).__init__()
        # Encoder
        self.down1 = DynamicFusionConvAttnBlock(in_channels, features[0], kernel_size=3, stride=1, padding=1, pool_size=pool_size, ablation_on_qk_channels=ablation_on_qk_channels)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.down2 = DynamicFusionConvAttnBlock(features[0], features[1], kernel_size=3, stride=1, padding=1, pool_size=pool_size, ablation_on_qk_channels=ablation_on_qk_channels)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.down3 = DynamicFusionConvAttnBlock(features[1], features[2], kernel_size=3, stride=1, padding=1, pool_size=pool_size, ablation_on_qk_channels=ablation_on_qk_channels)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.down4 = DynamicFusionConvAttnBlock(features[2], features[3], kernel_size=3, stride=1, padding=1, pool_size=pool_size, ablation_on_qk_channels=ablation_on_qk_channels)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Bottleneck
        self.bottleneck = DynamicFusionConvAttnBlock(features[3], features[3]*2, kernel_size=3, stride=1, padding=1, pool_size=pool_size, ablation_on_qk_channels=ablation_on_qk_channels)
        
        # Decoder
        self.up4 = nn.ConvTranspose2d(features[3]*2, features[3], kernel_size=2, stride=2)
        self.up_conv4 = DynamicFusionConvAttnBlock(features[3]*2, features[3], kernel_size=3, stride=1, padding=1, pool_size=pool_size, ablation_on_qk_channels=ablation_on_qk_channels)
        
        self.up3 = nn.ConvTranspose2d(features[3], features[2], kernel_size=2, stride=2)
        self.up_conv3 = DynamicFusionConvAttnBlock(features[2]*2, features[2], kernel_size=3, stride=1, padding=1, pool_size=pool_size, ablation_on_qk_channels=ablation_on_qk_channels)
        
        self.up2 = nn.ConvTranspose2d(features[2], features[1], kernel_size=2, stride=2)
        self.up_conv2 = DynamicFusionConvAttnBlock(features[1]*2, features[1], kernel_size=3, stride=1, padding=1, pool_size=pool_size, ablation_on_qk_channels=ablation_on_qk_channels)
        
        self.up1 = nn.ConvTranspose2d(features[1], features[0], kernel_size=2, stride=2)
        self.up_conv1 = DynamicFusionConvAttnBlock(features[0]*2, features[0], kernel_size=3, stride=1, padding=1, pool_size=pool_size, ablation_on_qk_channels=ablation_on_qk_channels)
        
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
    
    def forward(self, x):
        # Encoder
        d1 = self.down1(x)            # [B, features[0], H, W]
        p1 = self.pool1(d1)
        
        d2 = self.down2(p1)           # [B, features[1], H/2, W/2]
        p2 = self.pool2(d2)
        
        d3 = self.down3(p2)           # [B, features[2], H/4, W/4]
        p3 = self.pool3(d3)
        
        d4 = self.down4(p3)           # [B, features[3], H/8, W/8]
        p4 = self.pool4(d4)
        
        # Bottleneck
        bn = self.bottleneck(p4)
        
        # Decoder
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
        
        out = self.final_conv(u1)
        return out

# 新增帶有殘差連接的 UNetDFCSA 類
class UNetDFCSARes(UNetDFCSA):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512], pool_size=8, ablation_on_qk_channels=8):
        """
        帶有殘差連接的 U-Net 模型，基於 DynamicFusionConvAttnBlock
        
        參數:
            in_channels: 輸入通道數
            out_channels: 輸出通道數
            features: 各層特徵通道數列表
            pool_size: 輕量自注意力模組下採樣後的解析度
        """
        super(UNetDFCSARes, self).__init__(in_channels, out_channels, features, pool_size, ablation_on_qk_channels)
        # 此類已繼承 UNetDFCSA 的所有功能
        # DynamicFusionConvAttnBlock 內部已包含殘差連接

if __name__ == "__main__":
    # 測試 DynamicFusionConvAttnBlock
    x = torch.randn(4, 3, 64, 64)
    block = DynamicFusionConvAttnBlock(in_channels=3, out_channels=64, pool_size=8, ablation_on_qk_channels=8)
    out = block(x)
    print("DynamicFusionConvAttnBlock 輸出形狀:", out.shape)
    
    # 測試 UNetDFCSARes
    model = UNetDFCSARes(in_channels=3, out_channels=1, features=[64, 128, 256, 512], pool_size=8, ablation_on_qk_channels=8)
    input_tensor = torch.randn(4, 3, 300, 300)
    output = model(input_tensor)
    print("UNetDFCSARes 輸出形狀:", output.shape)
