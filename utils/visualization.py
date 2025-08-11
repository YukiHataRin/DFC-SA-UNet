import os
import torch
import numpy as np
import matplotlib
# Set non-interactive backend to avoid thread issues
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import cv2
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import torchvision.transforms as transforms
import pandas as pd

def save_loss_plot(train_losses, val_losses, save_path):
    """
    Save training and validation loss curve
    
    Parameters:
        train_losses: Training loss list
        val_losses: Validation loss list
        save_path: Save path
    """
    # Ensure path uses correct slash format
    save_path = save_path.replace('\\', '/')
    
    # 保存損失曲線圖
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close('all')  # Ensure all figures are closed
    
    # 保存損失數據到 CSV
    csv_path = os.path.splitext(save_path)[0] + '.csv'
    epochs = list(range(1, len(train_losses) + 1))
    save_metrics_to_csv(epochs, train_losses, val_losses, 'Loss', csv_path)

def save_metrics_to_csv(epochs, train_metrics, val_metrics, metric_name, save_path):
    """
    將訓練和驗證指標保存到 CSV 文件
    
    參數:
        epochs: 訓練輪數列表
        train_metrics: 訓練指標列表
        val_metrics: 驗證指標列表
        metric_name: 指標名稱
        save_path: 保存路徑
    """
    # 確保路徑使用正確的斜線格式
    save_path = save_path.replace('\\', '/')
    
    # 創建 DataFrame
    df = pd.DataFrame({
        'Epoch': epochs,
        f'Train_{metric_name}': train_metrics,
        f'Val_{metric_name}': val_metrics
    })
    
    # 保存到 CSV
    df.to_csv(save_path, index=False)
    print(f"已保存 {metric_name} 指標到 CSV: {save_path}")

def save_metrics_plot(epochs, train_metrics, val_metrics, metric_name, save_path):
    """
    Save training and validation metric curve
    
    Parameters:
        epochs: Number of training epochs
        train_metrics: Training metric list
        val_metrics: Validation metric list
        metric_name: Metric name
        save_path: Save path
    """
    # Ensure path uses correct slash format
    save_path = save_path.replace('\\', '/')
    
    # 保存指標曲線圖
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_metrics, label=f'Train {metric_name}')
    plt.plot(epochs, val_metrics, label=f'Validation {metric_name}')
    plt.xlabel('Epoch')
    plt.ylabel(metric_name)
    plt.title(f'Training and Validation {metric_name}')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close('all')  # Ensure all figures are closed
    
    # 保存指標數據到 CSV
    csv_path = os.path.splitext(save_path)[0] + '.csv'
    save_metrics_to_csv(epochs, train_metrics, val_metrics, metric_name, csv_path)

def tensor_to_numpy(tensor):
    """
    Convert tensor to numpy array
    
    Parameters:
        tensor: Input tensor
    
    Returns:
        numpy array
    """
    if tensor.is_cuda:
        tensor = tensor.cpu()
    if tensor.requires_grad:
        tensor = tensor.detach()
    return tensor.numpy()

def create_overlay(image, mask, alpha=0.5):
    """
    Create overlay image
    
    Parameters:
        image: Original image (H, W, C) RGB format
        mask: Mask (H, W) value range [0, 1]
        alpha: Transparency
    
    Returns:
        Overlay image (RGB format)
    """
    # Ensure input image is in RGB format
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    
    # Ensure mask is binary
    if mask.max() > 1:
        mask = (mask > 128).astype(np.float32) / 255.0
    
    # Create red mask
    overlay = image.copy()
    mask_bool = mask > 0.5
    
    # Check if mask_bool contains any True values
    if np.any(mask_bool):
        # Apply mask directly to red channel
        overlay[mask_bool, 0] = int(255 * alpha + overlay[mask_bool, 0].mean() * (1 - alpha))  # Red channel
        overlay[mask_bool, 1] = int(overlay[mask_bool, 1].mean() * (1 - alpha))  # Green channel
        overlay[mask_bool, 2] = int(overlay[mask_bool, 2].mean() * (1 - alpha))  # Blue channel
    
    return overlay

def create_combined_visualization(img, pred, mask, filename, save_path):
    """
    Create combined visualization image, showing from left to right:
    Original image, prediction, ground truth mask, edge overlay, and prediction-ground truth overlay
    
    Parameters:
        img: Original image (H, W, C) RGB format
        pred: Prediction mask (H, W)
        mask: Ground truth mask (H, W)
        filename: Filename
        save_path: Save path
    """
    # Ensure path uses correct slash format and has extension
    save_path = save_path.replace('\\', '/')
    if not save_path.lower().endswith(('.png', '.jpg', '.jpeg')):
        save_path = save_path + '.png'
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Ensure input image is RGB format numpy array
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8)
    
    # Ensure mask and pred are binary
    mask_binary = (mask > 128).astype(np.uint8) * 255
    pred_binary = (pred > 128).astype(np.uint8) * 255
    
    # Convert numpy arrays to PIL images
    img_pil = Image.fromarray(img, mode='RGB')
    
    # Create colored mask images
    mask_rgb = np.zeros_like(img)
    mask_rgb[:,:,1] = mask_binary  # Green channel - ground truth mask
    mask_pil = Image.fromarray(mask_rgb, mode='RGB')
    
    pred_rgb = np.zeros_like(img)
    pred_rgb[:,:,0] = pred_binary  # Red channel - prediction mask
    pred_pil = Image.fromarray(pred_rgb, mode='RGB')
    
    # Create edge overlay
    # Use PIL's ImageFilter for edge detection
    mask_pil_gray = Image.fromarray(mask_binary, mode='L')
    pred_pil_gray = Image.fromarray(pred_binary, mode='L')
    
    # Create edge overlay image
    edge_overlay = img.copy()
    
    # Process edges using numpy
    mask_edge = np.array(mask_pil_gray.filter(ImageFilter.FIND_EDGES))
    pred_edge = np.array(pred_pil_gray.filter(ImageFilter.FIND_EDGES))
    
    # Set colors at edge positions
    if np.any(mask_edge > 0):
        edge_overlay[mask_edge > 0, 1] = 255  # Green channel - ground truth edges
        edge_overlay[mask_edge > 0, 0] = 0
        edge_overlay[mask_edge > 0, 2] = 0
    
    if np.any(pred_edge > 0):
        edge_overlay[pred_edge > 0, 0] = 255  # Red channel - prediction edges
        edge_overlay[pred_edge > 0, 1] = 0
        edge_overlay[pred_edge > 0, 2] = 0
    
    edge_overlay_pil = Image.fromarray(edge_overlay, mode='RGB')
    
    # Create prediction on ground truth overlay
    gt_pred_overlay = np.zeros_like(img)
    if np.any(mask_binary > 0):
        gt_pred_overlay[mask_binary > 0, 1] = 255  # Green for ground truth mask
    if np.any(pred_binary > 0):
        gt_pred_overlay[pred_binary > 0, 0] = 255  # Red for prediction area
    gt_pred_overlay_pil = Image.fromarray(gt_pred_overlay, mode='RGB')
    
    # Get image dimensions
    h, w = img_pil.size[1], img_pil.size[0]
    
    # Add title area and borders
    title_height = 40  # Title area height
    border_width = 3   # Border width
    
    # Create combined image, including title area
    combined_width = w * 5 + border_width * 4
    combined_height = h + title_height
    combined_img = Image.new('RGB', (combined_width, combined_height), color=(50, 50, 50))
    
    # Place images from left to right, adding borders
    # Original image
    combined_img.paste(img_pil, (0, title_height))
    
    # First border
    for y in range(title_height, combined_height):
        for x in range(w, w + border_width):
            combined_img.putpixel((x, y), (255, 255, 255))
    
    # Prediction
    combined_img.paste(pred_pil, (w + border_width, title_height))
    
    # Second border
    for y in range(title_height, combined_height):
        for x in range(2*w + border_width, 2*w + 2*border_width):
            combined_img.putpixel((x, y), (255, 255, 255))
    
    # Ground truth mask
    combined_img.paste(mask_pil, (2*w + 2*border_width, title_height))
    
    # Third border
    for y in range(title_height, combined_height):
        for x in range(3*w + 2*border_width, 3*w + 3*border_width):
            combined_img.putpixel((x, y), (255, 255, 255))
    
    # Edge overlay
    combined_img.paste(edge_overlay_pil, (3*w + 3*border_width, title_height))
    
    # Fourth border
    for y in range(title_height, combined_height):
        for x in range(4*w + 3*border_width, 4*w + 4*border_width):
            combined_img.putpixel((x, y), (255, 255, 255))
    
    # Prediction-ground truth overlay
    combined_img.paste(gt_pred_overlay_pil, (4*w + 4*border_width, title_height))
    
    # Add titles
    draw = ImageDraw.Draw(combined_img)
    try:
        # Try to load a font that supports Chinese characters
        font = ImageFont.truetype("simhei.ttf", 16)
    except IOError:
        # If unable to load Chinese font, use default font
        font = ImageFont.load_default()
    
    titles = ["Original", "Prediction", "Ground Truth", "Edge Overlay", "Pred-GT Overlay"]
    
    for i, title in enumerate(titles):
        # Calculate title position, considering borders
        offset = i * (w + border_width) + (border_width * (i > 0))
        
        # Use textbbox instead of textsize (for newer versions of Pillow)
        try:
            # First try using textbbox method
            text_bbox = draw.textbbox((0, 0), title, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
        except AttributeError:
            # If textbbox is not available, try using textlength method
            try:
                text_width = draw.textlength(title, font=font)
                # Estimate text height
                text_height = font.getsize(title)[1]
            except (AttributeError, TypeError):
                # If both are unavailable, use the old getsize method
                text_width, text_height = font.getsize(title)
        
        text_x = offset + (w - text_width) // 2
        text_y = (title_height - text_height) // 2
        
        draw.text((text_x, text_y), title, fill=(255, 255, 255), font=font)
    
    # Add horizontal line at the bottom of the title area
    for x in range(combined_width):
        combined_img.putpixel((x, title_height-1), (255, 255, 255))
        combined_img.putpixel((x, title_height), (255, 255, 255))
    
    # Save combined image
    combined_img.save(save_path)
    
    return np.array(combined_img)

def save_prediction_samples(images, predictions, masks, filenames, save_dir):
    """
    保存預測樣本
    
    參數:
        images: 圖像批次 (B, C, H, W)
        predictions: 預測結果批次 (B, 1, H, W)
        masks: 真實標籤批次 (B, 1, H, W)
        filenames: 文件名列表
        save_dir: 保存目錄
    """
    # 確保主目錄存在
    os.makedirs(save_dir, exist_ok=True)
    
    # 轉換張量為 numpy 數組
    images = tensor_to_numpy(images)
    predictions = tensor_to_numpy(predictions)
    masks = tensor_to_numpy(masks)
    
    # 定義反歸一化轉換
    def denormalize(img):
        """反歸一化圖像"""
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = img * std[:, None, None] + mean[:, None, None]  # 反歸一化
        img = np.clip(img, 0, 1)  # 裁剪到 [0,1] 範圍
        img = (img * 255).astype(np.uint8)  # 轉換到 [0,255]
        return img
    
    # 處理每個樣本
    for i in range(len(filenames)):
        # 獲取當前樣本
        img = images[i]  # 保持 CHW 格式
        pred = predictions[i, 0]  # 移除通道維度
        mask = masks[i, 0]  # 移除通道維度
        
        # 反歸一化圖像
        img = denormalize(img)
        img = img.transpose(1, 2, 0)  # CHW -> HWC
        
        # 確保遮罩為二值
        pred = (pred > 0.5).astype(np.uint8) * 255
        mask = (mask > 0.5).astype(np.uint8) * 255
        
        # 創建基礎文件名
        base_filename = os.path.splitext(filenames[i])[0]
        
        # 創建樣本子目錄
        sample_dir = os.path.join(save_dir, base_filename)
        os.makedirs(sample_dir, exist_ok=True)
        
        # 創建保存路徑
        img_save_path = os.path.join(sample_dir, "original.png")
        pred_save_path = os.path.join(sample_dir, "prediction.png")
        mask_save_path = os.path.join(sample_dir, "ground_truth.png")
        overlay_save_path = os.path.join(sample_dir, "overlay.png")
        
        # combined 圖片保存在主目錄
        combined_save_path = os.path.join(save_dir, f"{base_filename}.png")
        
        # 保存原始圖像
        cv2.imwrite(img_save_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        
        # 保存預測遮罩
        cv2.imwrite(pred_save_path, pred)
        
        # 保存真實標籤
        cv2.imwrite(mask_save_path, mask)
        
        # 創建並保存疊加圖
        overlay = create_overlay(img, pred / 255.0)
        cv2.imwrite(overlay_save_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        
        # 創建並保存組合可視化圖
        create_combined_visualization(img, pred, mask, base_filename, combined_save_path) 