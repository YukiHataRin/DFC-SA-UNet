import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchEmbedding(nn.Module):
    """將影像轉換為 Patch Embeddings"""
    def __init__(self, img_dim, patch_dim, in_channels, embed_dim):
        super().__init__()
        self.img_dim = img_dim
        self.patch_dim = patch_dim
        self.num_patches = (img_dim // patch_dim) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_dim, stride=patch_dim)

    def forward(self, x):
        # x: (B, C, H, W)
        x = self.proj(x)  # (B, embed_dim, num_patches_h, num_patches_w)
        x = x.flatten(2)  # (B, embed_dim, num_patches)
        x = x.transpose(1, 2)  # (B, num_patches, embed_dim)
        return x

class VisionTransformerForSegmentation(nn.Module):
    def __init__(self, *,
                 img_dim=224,
                 patch_dim=16,
                 in_channels=3,
                 num_classes=1, # 通常分割任務的類別數 (例如二元分割為1)
                 embed_dim=768,
                 num_layers=12,
                 num_heads=12,
                 mlp_dim=3072,
                 dropout=0.1,
                 segmentation_head_upsample_layers=4 # 分割頭的上採樣層數
                ):
        super(VisionTransformerForSegmentation, self).__init__()

        self.img_dim = img_dim
        self.patch_dim = patch_dim
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.embed_dim = embed_dim

        # 1. Patch Embedding
        self.patch_embed = PatchEmbedding(img_dim, patch_dim, in_channels, embed_dim)
        self.num_patches = self.patch_embed.num_patches

        # 2. Class token (可選，對於純分割ViT有時不使用，這裡我們先不加，直接用patch tokens)
        # self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        # 3. Positional Embedding
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, embed_dim)) # 如果有cls token要 self.num_patches + 1
        self.pos_drop = nn.Dropout(dropout)

        # 4. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            activation='gelu', # ViT 通常使用 gelu
            batch_first=True # 輸入格式為 (batch, seq, feature)
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 5. Segmentation Head
        # 計算 Transformer 輸出後特徵圖的維度 (H_feat, W_feat)
        self.features_h = img_dim // patch_dim
        self.features_w = img_dim // patch_dim

        # 分割頭：一系列上採樣卷積層
        # 目標是將 (B, num_patches, embed_dim) -> (B, num_classes, H, W)
        # 首先將 embed_dim 轉換為較小的通道數，例如 256 或 embed_dim // 2
        # 這裡我們直接使用 embed_dim，但實務上可能會先降維
        
        # 我們需要從 (B, num_patches, embed_dim) -> (B, embed_dim, H_feat, W_feat)
        # 然後進行上採樣
        
        # 簡易分割頭：這裡用 ConvTranspose2d 進行上採樣
        # 你可以設計更複雜的分割頭，例如類似 UNet 的解碼器結構
        
        seg_head_layers = []
        current_channels = embed_dim
        
        # 根據 patch_dim 決定上採樣的 scale_factor
        # 例如，如果 patch_dim 是 16，我們需要放大 16 倍回到原始尺寸
        # 我們可以用多個 scale_factor=2 的 ConvTranspose2d 層
        
        # 確保最終的 scale_factor 正確
        total_scale_factor = 1
        
        # 上採樣層，目標是放大 patch_dim 倍
        # 這裡使用固定的上採樣層數，每層放大2倍
        # 你可以根據 patch_dim 和 img_dim 動態調整層數和 scale_factor
        
        # 決定分割頭中的初始通道數
        # 可以是 embed_dim，或者為了減少計算量先降維
        # 這裡我們假設先有一個 Conv2d 將 embed_dim 映射到一個中間通道數
        # 或是直接用 embed_dim 開始上採樣
        
        # 這裡我們用一個簡單的例子，逐步上採樣
        # 假設 patch_dim 是 16，我們需要放大 16 倍 (2^4)
        # 如果 segmentation_head_upsample_layers = 4，每層放大 2 倍
        
        # 如果 patch_dim 不是 2 的次方，這個簡易頭可能需要調整
        
        upsample_scale = 2 # 每層上採樣的倍數
        
        for i in range(segmentation_head_upsample_layers):
            out_channels_layer = current_channels // 2 if i < segmentation_head_upsample_layers -1 else current_channels // 2 # 可以自行調整
            if out_channels_layer < num_classes * 4 and i < segmentation_head_upsample_layers - 1 : # 避免通道數過少
                 out_channels_layer = num_classes * 4 if num_classes * 4 < current_channels else current_channels // 2

            seg_head_layers.append(
                nn.ConvTranspose2d(
                    in_channels=current_channels,
                    out_channels=out_channels_layer,
                    kernel_size=upsample_scale*2, # kernel_size 設為 scale_factor 的兩倍避免棋盤效應
                    stride=upsample_scale,
                    padding=upsample_scale // 2 # padding 保持尺寸
                )
            )
            seg_head_layers.append(nn.BatchNorm2d(out_channels_layer))
            seg_head_layers.append(nn.ReLU(inplace=True))
            current_channels = out_channels_layer
            total_scale_factor *= upsample_scale

        # 最後一個卷積層，將通道數調整到 num_classes
        # 確保上採樣倍數與 patch_dim 匹配
        if total_scale_factor != patch_dim:
             print(f"Warning: Segmentation head total_scale_factor ({total_scale_factor}) "
                   f"does not match patch_dim ({patch_dim}). "
                   f"Output size might not match input image size perfectly. "
                   f"Consider adjusting segmentation_head_upsample_layers.")

        seg_head_layers.append(
            nn.Conv2d(current_channels, num_classes, kernel_size=1)
        )
        
        self.segmentation_head = nn.Sequential(*seg_head_layers)


    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_dim and W == self.img_dim, \
            f"Input image size ({H}x{W}) doesn't match model ({self.img_dim}x{self.img_dim})."

        # 1. Patch Embedding
        x = self.patch_embed(x)  # (B, num_patches, embed_dim)

        # 2. Positional Embedding (如果使用了 cls_token 要調整)
        # tokens = torch.cat((self.cls_token.expand(B, -1, -1), x), dim=1)
        # x = x + self.pos_embed.expand(B, -1, -1) # 如果沒有 cls token
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # 3. Transformer Encoder
        x_encoded = self.transformer_encoder(x)  # (B, num_patches, embed_dim)

        # 4. Segmentation Head
        # 將 Transformer Encoder 的輸出 reshape 回 2D 特徵圖
        # x_encoded: (B, num_patches, embed_dim)
        # 我們需要 (B, embed_dim, H_feat, W_feat)
        x_reshaped = x_encoded.transpose(1, 2).reshape(
            B, self.embed_dim, self.features_h, self.features_w
        )
        
        # 通過分割頭得到分割遮罩
        logits = self.segmentation_head(x_reshaped) # (B, num_classes, H, W)
        
        # 確保輸出尺寸與輸入一致 (如果上採樣層設計得當，應該會一致)
        # 如果不一致，可以考慮使用 F.interpolate
        if logits.shape[2:] != (H, W):
            logits = F.interpolate(logits, size=(H,W), mode='bilinear', align_corners=False)
            
        return logits

if __name__ == '__main__':
    # 測試模型
    batch_size = 2
    img_dim = 224 # 必須是 patch_dim 的整數倍
    patch_dim = 16
    in_channels = 3
    num_classes = 1 # 二元分割

    # 計算 segmentation_head_upsample_layers
    # 我們希望總的上採樣倍數是 patch_dim
    # 每層上採樣2倍，所以需要 log2(patch_dim) 層
    import math
    segmentation_head_upsample_layers = int(math.log2(patch_dim))


    model = VisionTransformerForSegmentation(
        img_dim=img_dim,
        patch_dim=patch_dim,
        in_channels=in_channels,
        num_classes=num_classes,
        embed_dim=768, # ViT-Base
        num_layers=12, # ViT-Base
        num_heads=12,  # ViT-Base
        mlp_dim=3072,  # ViT-Base
        dropout=0.1,
        segmentation_head_upsample_layers=segmentation_head_upsample_layers
    )

    dummy_input = torch.randn(batch_size, in_channels, img_dim, img_dim)
    
    print(f"Input shape: {dummy_input.shape}")
    
    # 計算模型參數量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    try:
        output = model(dummy_input)
        print(f"Output shape: {output.shape}")
        assert output.shape == (batch_size, num_classes, img_dim, img_dim), "Output shape mismatch!"
        print("Model forward pass successful!")
    except Exception as e:
        print(f"Error during model forward pass: {e}")
        import traceback
        traceback.print_exc()

    # 嘗試使用 torchinfo 打印模型摘要
    try:
        from torchinfo import summary
        summary(model, input_size=(batch_size, in_channels, img_dim, img_dim), device="cpu")
    except ImportError:
        print("\nInstall torchinfo for model summary: pip install torchinfo")
    except Exception as e:
        print(f"\nError generating model summary with torchinfo: {e}")