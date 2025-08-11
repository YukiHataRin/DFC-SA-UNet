import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import matplotlib
# 設置使用非互動式後端，避免執行緒問題
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import yaml
import time
from datetime import datetime

from utils.metrics import dice_loss, tversky_loss, calculate_metrics
from utils.visualization import save_loss_plot, save_metrics_plot, save_prediction_samples

class Trainer:
    def __init__(self, model, train_loader, val_loader, optimizer, device, config):
        """
        訓練器類
        
        參數:
            model: 模型
            train_loader: 訓練數據加載器
            val_loader: 驗證數據加載器
            optimizer: 優化器
            device: 訓練設備
            config: 配置對象
        """
        self.config = config
        self.device = device
        self.model = model.to(device)
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # 設置損失函數
        self.loss_type = self.config['training'].get('loss', {}).get('type', 'dice')
        self.loss_params = self.config['training'].get('loss', {}).get('params', {})
        
        print(f"使用損失函數: {self.loss_type}")
        if self.loss_type == 'tversky':
            alpha = self.loss_params.get('alpha', 0.5)
            beta = self.loss_params.get('beta', 0.5)
            print(f"Tversky 損失參數: alpha={alpha}, beta={beta}")
        elif self.loss_type == 'bce_dice':
            weight_bce = self.loss_params.get('weight_bce', 1.0)
            weight_dice = self.loss_params.get('weight_dice', 1.0)
            print(f"BCE+Dice 損失參數: weight_bce={weight_bce}, weight_dice={weight_dice}")
        elif self.loss_type == 'joint':
            bce_weight = self.loss_params.get('bce_weight', 1.0)
            dice_weight = self.loss_params.get('dice_weight', 1.0)
            contour_weight = self.loss_params.get('contour_weight', 1.0)
            print(f"Joint 損失參數: bce_weight={bce_weight}, dice_weight={dice_weight}, contour_weight={contour_weight}")
        
        # 初始化記錄器
        self.train_losses = []
        self.val_losses = []
        self.train_dice_scores = []
        self.val_dice_scores = []
        self.train_iou_scores = []
        self.val_iou_scores = []
        self.epochs = []
        
        # 創建日誌目錄
        self.log_dir = self._normalize_path(self.config['logging']['log_dir'])
        self.images_dir = self._normalize_path(self.config['logging']['images_dir'])
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.images_dir, exist_ok=True)
        
        # 設置最佳模型保存路徑
        self.best_model_path = os.path.join(self.log_dir, 'best_model.pth').replace('\\', '/')
        self.best_val_loss = float('inf')
        
        # 設置checkpoint目錄
        self.checkpoint_dir = os.path.join(self.log_dir, 'checkpoints').replace('\\', '/')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # 記錄開始時間
        self.start_time = time.time()
        
        print(f"模型將在 {self.device} 上訓練")
        print(f"日誌將保存在 {self.log_dir}")
        print(f"檢查點將保存在 {self.checkpoint_dir}")
        
        # 設置訓練參數
        self.num_epochs = self.config['training']['num_epochs']
    
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
    
    def train_epoch(self, epoch):
        """訓練一個 epoch"""
        self.model.train()
        running_loss = 0.0
        running_iou = 0.0
        running_dice = 0.0
        
        # 使用 tqdm 顯示進度條
        progress_bar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.num_epochs} [Train]')
        
        for batch_idx, batch in enumerate(progress_bar):
            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)
            
            # 前向傳播
            self.optimizer.zero_grad()
            outputs = self.model(images)
            
            # 應用 sigmoid 激活函數
            outputs = torch.sigmoid(outputs)
            
            # 計算損失和指標
            loss_type = self.config['training']['loss']['type']
            loss_params = self.config['training']['loss']['params']
            
            metrics = calculate_metrics(outputs, masks, loss_type, loss_params)
            loss = metrics['loss']
            
            # 檢查 loss 是否為 NaN
            if torch.isnan(loss).any():
                print(f"Warning: NaN loss detected at batch {batch_idx}")
                print(f"  outputs range: [{outputs.min():.6f}, {outputs.max():.6f}]")
                print(f"  masks range: [{masks.min():.6f}, {masks.max():.6f}]")
                print("  Skipping this batch...")
                continue
            
            # 檢查 loss 是否過大
            if loss.item() > 100:
                print(f"Warning: Very large loss detected: {loss.item():.6f} at batch {batch_idx}")
            
            # 反向傳播和優化
            loss.backward()
            
            # 梯度裁剪以防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # 更新統計信息
            running_loss += loss.item()
            running_iou += metrics['iou']
            running_dice += metrics['dice']
            
            # 更新進度條
            progress_bar.set_postfix({
                'loss': running_loss / (batch_idx + 1),
                'iou': running_iou / (batch_idx + 1),
                'dice': running_dice / (batch_idx + 1)
            })
        
        # 計算平均指標
        epoch_loss = running_loss / len(self.train_loader)
        epoch_iou = running_iou / len(self.train_loader)
        epoch_dice = running_dice / len(self.train_loader)
        
        return epoch_loss, epoch_iou, epoch_dice
    
    def validate_epoch(self, dataloader):
        """
        驗證一個epoch
        
        參數:
            dataloader: 驗證數據加載器
        
        返回:
            包含驗證指標的字典
        """
        self.model.eval()
        running_loss = 0.0
        running_iou = 0.0
        running_dice = 0.0
        
        # 使用 tqdm 顯示進度條
        progress_bar = tqdm(dataloader, desc=f'Validation')
        
        # 存儲每個樣本的指標，用於找出最好和最差的樣本
        sample_metrics = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(progress_bar):
                images = batch['image'].to(self.device)
                masks = batch['mask'].to(self.device)
                filenames = batch['filename']
                
                # 前向傳播
                outputs = self.model(images)
                
                # 應用 sigmoid 激活函數
                outputs = torch.sigmoid(outputs)
                
                # 計算損失和指標
                loss_type = self.config['training']['loss']['type']
                loss_params = self.config['training']['loss']['params']
                
                metrics = calculate_metrics(outputs, masks, loss_type, loss_params)
                
                # 檢查 loss 是否為 NaN
                if torch.isnan(metrics['loss']).any():
                    print(f"Warning: NaN loss detected in validation at batch {batch_idx}")
                    continue
                
                # 更新統計信息
                running_loss += metrics['loss'].item()
                running_iou += metrics['iou']
                running_dice += metrics['dice']
                
                # 更新進度條
                progress_bar.set_postfix({
                    'loss': running_loss / (batch_idx + 1),
                    'iou': running_iou / (batch_idx + 1),
                    'dice': running_dice / (batch_idx + 1)
                })
                
                # 計算每個樣本的指標
                for i in range(images.size(0)):
                    sample_output = outputs[i:i+1]
                    sample_mask = masks[i:i+1]
                    sample_metric = calculate_metrics(sample_output, sample_mask, loss_type, loss_params)
                    sample_metrics.append({
                        'batch_idx': batch_idx,
                        'sample_idx': i,
                        'image': images[i].cpu(),
                        'mask': masks[i].cpu(),
                        'output': outputs[i].cpu(),
                        'filename': filenames[i],
                        'metrics': {
                            'loss': sample_metric['loss'].item(),
                            'iou': sample_metric['iou'],
                            'dice': sample_metric['dice']
                        }
                    })
        
        # 計算平均指標
        epoch_loss = running_loss / len(dataloader)
        epoch_iou = running_iou / len(dataloader)
        epoch_dice = running_dice / len(dataloader)
        
        # 根據 Dice 係數排序樣本
        sample_metrics.sort(key=lambda x: x['metrics']['dice'])
        
        # 獲取最差和最佳樣本
        worst_samples = sample_metrics[:self.config['logging']['save_best_worst_samples']]
        best_samples = sample_metrics[-self.config['logging']['save_best_worst_samples']:]
        
        return {
            'loss': epoch_loss,
            'iou': epoch_iou,
            'dice': epoch_dice,
            'best_samples': best_samples,
            'worst_samples': worst_samples
        }
    
    def save_checkpoint(self, epoch, metrics, is_best=False):
        """
        保存檢查點
        
        參數:
            epoch: 當前epoch
            metrics: 指標字典
            is_best: 是否為最佳模型
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_dice_scores': self.train_dice_scores,
            'val_dice_scores': self.val_dice_scores,
            'train_iou_scores': self.train_iou_scores,
            'val_iou_scores': self.val_iou_scores,
            'best_val_loss': self.best_val_loss,
            'metrics': metrics
        }
        
        # 保存最新檢查點
        checkpoint_path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth').replace('\\', '/')
        torch.save(checkpoint, checkpoint_path)
        
        # 如果是最佳模型，則另外保存一份
        if is_best:
            torch.save(self.model.state_dict(), self.best_model_path)
            best_checkpoint_path = os.path.join(self.checkpoint_dir, 'best_checkpoint.pth').replace('\\', '/')
            torch.save(checkpoint, best_checkpoint_path)
    
    def load_checkpoint(self, checkpoint_path):
        """
        加載檢查點
        
        參數:
            checkpoint_path: 檢查點路徑
        
        返回:
            加載的epoch
        """
        # 確保路徑使用正確的斜線格式
        checkpoint_path = checkpoint_path.replace('\\', '/')
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.train_dice_scores = checkpoint['train_dice_scores']
        self.val_dice_scores = checkpoint['val_dice_scores']
        self.train_iou_scores = checkpoint['train_iou_scores']
        self.val_iou_scores = checkpoint['val_iou_scores']
        self.best_val_loss = checkpoint['best_val_loss']
        
        return checkpoint['epoch']
    
    def train(self, resume_from=None):
        """
        訓練模型
        
        參數:
            resume_from: 恢復訓練的檢查點路徑
        """
        # 獲取訓練參數
        start_epoch = 0
        
        # 如果指定了檢查點，則從檢查點恢復訓練
        if resume_from:
            checkpoint = self.load_checkpoint(resume_from)
            start_epoch = checkpoint['epoch'] + 1
            print(f"從 epoch {start_epoch} 恢復訓練")
        
        # 初始化指標記錄
        self.epochs = []
        self.train_losses = []
        self.val_losses = []
        self.train_dice_scores = []
        self.val_dice_scores = []
        self.train_iou_scores = []
        self.val_iou_scores = []
        
        # 訓練循環
        best_val_dice = 0.0
        
        for epoch in range(start_epoch, self.num_epochs):
            # 訓練
            train_metrics = self.train_epoch(epoch)
            
            # 驗證
            val_results = self.validate_epoch(self.val_loader)
            
            # 記錄指標
            self.epochs.append(epoch + 1)
            self.train_losses.append(train_metrics[0])
            self.val_losses.append(val_results['loss'])
            self.train_dice_scores.append(train_metrics[2])
            self.val_dice_scores.append(val_results['dice'])
            self.train_iou_scores.append(train_metrics[1])
            self.val_iou_scores.append(val_results['iou'])
            
            # 打印指標
            print(f"Epoch [{epoch+1}/{self.num_epochs}]")
            print(f"  Train Loss: {train_metrics[0]:.4f}, Dice: {train_metrics[2]:.4f}, IoU: {train_metrics[1]:.4f}")
            print(f"  Val Loss: {val_results['loss']:.4f}, Dice: {val_results['dice']:.4f}, IoU: {val_results['iou']:.4f}")
            
            # 檢查是否為最佳模型
            is_best = val_results['dice'] > best_val_dice
            if is_best:
                best_val_dice = val_results['dice']
                print(f"  Saved best model with validation dice: {best_val_dice:.4f}")
            
            # 保存檢查點
            if (epoch + 1) % self.config['training']['save_checkpoint_freq'] == 0 or is_best:
                self.save_checkpoint(epoch, val_results, is_best)
                if (epoch + 1) % self.config['training']['save_checkpoint_freq'] == 0:
                    print(f"  Saved checkpoint at epoch {epoch+1}")
            
            # 保存損失和指標圖
            save_loss_plot(
                self.train_losses, 
                self.val_losses, 
                os.path.join(self.images_dir, 'loss_plot.png').replace('\\', '/')
            )
            
            save_metrics_plot(
                self.epochs,
                self.train_dice_scores,
                self.val_dice_scores,
                'Dice',
                os.path.join(self.images_dir, 'dice_plot.png').replace('\\', '/')
            )
            
            save_metrics_plot(
                self.epochs,
                self.train_iou_scores,
                self.val_iou_scores,
                'IoU',
                os.path.join(self.images_dir, 'iou_plot.png').replace('\\', '/')
            )
            
            # 保存最佳和最差樣本
            if epoch % 1 == 0:  # 每個epoch保存
                epoch_dir = os.path.join(self.log_dir, f'epoch_{epoch+1}')
                os.makedirs(epoch_dir, exist_ok=True)
                
                # 保存最佳樣本
                best_samples_dir = os.path.join(epoch_dir, 'best_samples')
                os.makedirs(best_samples_dir, exist_ok=True)
                
                for i, sample in enumerate(val_results['best_samples']):
                    img = sample['image']
                    mask = sample['mask']
                    pred = sample['output']
                    dice = sample['metrics']['dice']
                    filename = sample['filename'].split('.')[0]  # 去除文件擴展名
                    
                    # 保存樣本
                    save_prediction_samples(
                        img.unsqueeze(0), 
                        pred.unsqueeze(0), 
                        mask.unsqueeze(0),
                        [filename], 
                        best_samples_dir
                    )
                
                # 保存最差樣本
                worst_samples_dir = os.path.join(epoch_dir, 'worst_samples')
                os.makedirs(worst_samples_dir, exist_ok=True)
                
                for i, sample in enumerate(val_results['worst_samples']):
                    img = sample['image']
                    mask = sample['mask']
                    pred = sample['output']
                    dice = sample['metrics']['dice']
                    filename = sample['filename'].split('.')[0]  # 去除文件擴展名
                    
                    # 保存樣本
                    save_prediction_samples(
                        img.unsqueeze(0), 
                        pred.unsqueeze(0), 
                        mask.unsqueeze(0),
                        [filename], 
                        worst_samples_dir
                    )
        
        # 計算總訓練時間
        total_time = time.time() - self.start_time
        hours, remainder = divmod(total_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        print(f"Training completed in {int(hours)}h {int(minutes)}m {int(seconds)}s")
        print(f"Best validation dice: {best_val_dice:.4f}")
        print(f"Best model saved to {self.best_model_path}") 