import os
import argparse
import yaml
import torch

from models.model_factory import ModelFactory
from utils.data_loader import DataLoaderFactory
from utils.trainer import Trainer

def normalize_path(path):
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

def main(config_path, resume_path=None, loss_type=None, alpha=None, beta=None, augmentation=None):
    """
    主函數
    
    參數:
        config_path: 配置文件路徑
        resume_path: 從檢查點恢復訓練
        loss_type: 損失函數類型 ('dice' 或 'tversky')
        alpha: Tversky Loss 的 alpha 參數
        beta: Tversky Loss 的 beta 參數
        augmentation: 是否啟用數據增強 (True 或 False)
    """
    # 標準化路徑
    config_path = normalize_path(config_path)
    if resume_path:
        resume_path = normalize_path(resume_path)
    
    # 加載配置
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 更新配置中的參數（如果有指定）
    if loss_type:
        config['training']['loss']['type'] = loss_type
    if alpha is not None:
        config['training']['loss']['params']['alpha'] = alpha
    if beta is not None:
        config['training']['loss']['params']['beta'] = beta
    if augmentation is not None:
        config['dataset']['augmentation'] = augmentation
    
    # 設置設備
    if config['training']['device'] == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(config['training']['device'])
    
    print(f'使用設備: {device}')
    
    # 創建數據加載器
    data_loader_factory = DataLoaderFactory(config)
    train_loader = data_loader_factory.get_train_loader()
    val_loader = data_loader_factory.get_val_loader()
    
    # 創建模型
    model = ModelFactory.get_model(config)
    model = model.to(device)
    
    # 創建優化器，確保參數類型正確
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=float(config['training'].get('learning_rate', 0.01)),
        momentum=float(config['training'].get('momentum', 0.9)),
        weight_decay=float(config['training'].get('weight_decay', 1e-4))
    )
    
    # 創建訓練器
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        config=config
    )
    
    # 如果指定了恢復路徑，載入檢查點
    if resume_path:
        print(f'從檢查點恢復訓練: {resume_path}')
        trainer.load_checkpoint(resume_path)
    
    # 開始訓練
    trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train segmentation model")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    parser.add_argument("--resume", type=str, help="Path to checkpoint to resume training from")
    parser.add_argument("--loss", type=str, choices=['dice', 'tversky', 'bce_dice', 'joint'], help="Loss function to use (dice, tversky, bce_dice, joint)")
    parser.add_argument("--alpha", type=float, help="Alpha parameter for Tversky loss (weight for false positives)")
    parser.add_argument("--beta", type=float, help="Beta parameter for Tversky loss (weight for false negatives)")
    parser.add_argument("--weight_bce", type=float, help="BCE weight for BCE+Dice loss")
    parser.add_argument("--weight_dice", type=float, help="Dice weight for BCE+Dice loss")
    parser.add_argument("--bce_weight", type=float, help="BCE weight for Joint loss")
    parser.add_argument("--dice_weight", type=float, help="Dice weight for Joint loss")
    parser.add_argument("--contour_weight", type=float, help="Contour weight for Joint loss")
    parser.add_argument("--augmentation", type=lambda x: (str(x).lower() == 'true'), 
                        help="Enable or disable data augmentation (true/false)")
    args = parser.parse_args()
    
    # 加載配置文件
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 更新配置中的損失函數設置
    if args.loss is not None:
        config['training']['loss']['type'] = args.loss
    if args.alpha is not None:
        config['training']['loss']['params']['alpha'] = args.alpha
    if args.beta is not None:
        config['training']['loss']['params']['beta'] = args.beta
    if args.weight_bce is not None:
        config['training']['loss']['params']['weight_bce'] = args.weight_bce
    if args.weight_dice is not None:
        config['training']['loss']['params']['weight_dice'] = args.weight_dice
    if args.bce_weight is not None:
        config['training']['loss']['params']['bce_weight'] = args.bce_weight
    if args.dice_weight is not None:
        config['training']['loss']['params']['dice_weight'] = args.dice_weight
    if args.contour_weight is not None:
        config['training']['loss']['params']['contour_weight'] = args.contour_weight
    
    main(args.config, args.resume, args.loss, args.alpha, args.beta, args.augmentation) 