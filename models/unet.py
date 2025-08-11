import torch
import torch.nn as nn
import torch.nn.functional as F

# 定義兩層卷積區塊
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

# 定義向下取樣區塊 (MaxPool2d 後接 DoubleConv)
class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        # 使用 ceil_mode=True 可處理非 2 的整數倍尺寸，但可能導致尺寸略有不同
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2, ceil_mode=True),
            DoubleConv(in_channels, out_channels)
        )
    def forward(self, x):
        return self.mpconv(x)

# 定義向上取樣區塊，修改 crop 邏輯，當尺寸不匹配時自動裁剪較大者
class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        
        # 修正：考慮到拼接後的通道數
        self.conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x1, x2):
        x1 = self.up(x1)
        # 計算尺寸差異
        diffY = x2.size(2) - x1.size(2)
        diffX = x2.size(3) - x1.size(3)
        # 當上採樣後的尺寸大於 encoder 的特徵，則裁剪 x1
        if diffY < 0 or diffX < 0:
            x1 = x1[:, :, :x2.size(2), :x2.size(3)]
        else:
            # 否則裁剪 encoder 的特徵
            x2 = x2[:, :, diffY // 2 : diffY // 2 + x1.size(2),
                      diffX // 2 : diffX // 2 + x1.size(3)]
        # 拼接兩個特徵圖 (在 channel 維度上)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

# 定義最後的輸出層，透過 1x1 卷積得到預期通道數
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    def forward(self, x):
        return self.conv(x)

# 定義完整的 UNet 結構
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        factor = 2 if bilinear else 1
        
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024 // factor)
        
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)
    
    def forward(self, x):
        x1 = self.inc(x)      # 輸出尺寸：300x300 (假設輸入為 300x300)
        x2 = self.down1(x1)   # 約 150x150
        x3 = self.down2(x2)   # 約 75x75
        x4 = self.down3(x3)   # 約 38x38 (ceil_mode 調整後)
        x5 = self.down4(x4)   # 約 19x19
        x = self.up1(x5, x4)  # 上採樣後與 x4 拼接
        x = self.up2(x, x3)   # 上採樣後與 x3 拼接
        x = self.up3(x, x2)   # 上採樣後與 x2 拼接
        x = self.up4(x, x1)   # 上採樣後與 x1 拼接
        logits = self.outc(x)
        return logits

# 測試模型，並計算 BCELoss
if __name__ == '__main__':
    # 假設輸入為 300x300 的 RGB 影像
    model = UNet(n_channels=3, n_classes=1)
    input_tensor = torch.randn(1, 3, 300, 300)
    output = model(input_tensor)
    print("模型輸出尺寸：", output.shape)  # 預期輸出尺寸約為 [1, 1, 300, 300]
    
    # 使用 BCELoss 進行訓練（注意：BCELoss 需對 logits 做 sigmoid 處理）
    criterion = nn.BCELoss()
    pred = torch.sigmoid(output)
    target = torch.rand_like(pred)  # 假設目標標籤
    loss = criterion(pred, target)
    print("損失值：", loss.item())
