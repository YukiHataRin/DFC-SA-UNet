import os
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from datasets.segmentation_dataset import SegmentationDataset

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
        self.img_size = tuple(config['dataset']['img_size'])
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
            transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            # 驗證集或不使用數據增強的訓練集
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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