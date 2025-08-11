import os
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from datasets.segmentation_dataset import SegmentationDataset

# 定義自訂數據增強類，類似於ExtCompose, ExtResize, 等
class ExtTransform:
    """基礎轉換類，應用於圖像和遮罩"""
    def __call__(self, img, mask):
        return img, mask

class ExtCompose(ExtTransform):
    """組合多個轉換"""
    def __init__(self, transforms):
        self.transforms = transforms
        
    def __call__(self, img, mask):
        for t in self.transforms:
            img, mask = t(img, mask)
        return img, mask

class ExtResize(ExtTransform):
    """調整圖像和遮罩大小"""
    def __init__(self, size):
        self.size = size
        
    def __call__(self, img, mask):
        img = img.resize(self.size, Image.BILINEAR)
        mask = mask.resize(self.size, Image.NEAREST)
        return img, mask

class ExtRandomRotation(ExtTransform):
    """隨機旋轉圖像和遮罩"""
    def __init__(self, degrees):
        self.degrees = degrees
        
    def __call__(self, img, mask):
        if np.random.random() < 0.5:
            angle = np.random.uniform(-self.degrees, self.degrees)
            img = img.rotate(angle, Image.BILINEAR)
            mask = mask.rotate(angle, Image.NEAREST)
        return img, mask

class ExtRandomHorizontalFlip(ExtTransform):
    """隨機水平翻轉圖像和遮罩"""
    def __call__(self, img, mask):
        if np.random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        return img, mask

class ExtToTensor(ExtTransform):
    """將圖像和遮罩轉換為tensor"""
    def __call__(self, img, mask):
        img = transforms.ToTensor()(img)
        mask = torch.from_numpy(np.array(mask, dtype=np.uint8)).float()
        mask = mask.unsqueeze(0) / 255.0  # 歸一化並添加通道維度
        mask = (mask > 0.5).float()  # 二值化處理
        return img, mask

class ExtNormalize(ExtTransform):
    """標準化圖像"""
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = mean
        self.std = std
        self.normalize = transforms.Normalize(mean=mean, std=std)
        
    def __call__(self, img, mask):
        img = self.normalize(img)
        return img, mask

class DataLoaderFactory:
    def __init__(self, config):
        """
        數據加載器工廠
        
        參數:
            config: 配置字典
        """
        self.config = config
        # 確保路徑使用正確的斜線格式
        self.train_dir = self._normalize_path(config['dataset']['train_dir'])
        self.val_dir = self._normalize_path(config['dataset']['val_dir'])
        self.batch_size = config['training']['batch_size']
        self.num_workers = config['training']['num_workers']
        self.img_size = tuple(config['dataset'].get('img_size', [224, 224]))  # 默認使用 224x224
        # 從配置文件中讀取是否使用數據增強
        self.use_augmentation = config['dataset']['augmentation']
        
        print(f"數據增強: {'啟用' if self.use_augmentation else '禁用'}")
    
    def _normalize_path(self, path):
        """
        標準化路徑，確保使用正確的斜線格式
        
        參數:
            path: 原始路徑
        
        返回:
            標準化後的路徑
        """
        # 將Windows風格的反斜線轉換為正斜線
        normalized_path = path.replace('\\', '/')
        return normalized_path
    
    def get_transforms(self, is_train=True):
        """
        獲取數據轉換
        
        參數:
            is_train: 是否為訓練集
        
        返回:
            數據轉換
        """
        if is_train and self.use_augmentation:
            # 訓練集使用數據增強
            print("使用數據增強進行訓練")
            transform = ExtCompose([
                ExtResize(self.img_size),
                ExtRandomRotation(degrees=90),  # 隨機90度旋轉
                ExtRandomHorizontalFlip(),       # 隨機水平翻轉
                ExtToTensor(),                   # 轉換為Tensor
                ExtNormalize()                   # 標準化
            ])
        else:
            # 驗證集或不使用數據增強的訓練集
            transform = ExtCompose([
                ExtResize(self.img_size),
                ExtToTensor(),
                ExtNormalize()
            ])
        
        return transform
    
    def get_train_loader(self):
        """
        獲取訓練數據加載器
        
        返回:
            訓練數據加載器
        """
        transform = self.get_transforms(is_train=True)
        train_dataset = SegmentationDataset(
            root=self.train_dir,
            transform=transform,
            img_size=self.img_size
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        return train_loader
    
    def get_val_loader(self):
        """
        獲取驗證數據加載器
        
        返回:
            驗證數據加載器
        """
        transform = self.get_transforms(is_train=False)
        val_dataset = SegmentationDataset(
            root=self.val_dir,
            transform=transform,
            img_size=self.img_size
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        return val_loader 