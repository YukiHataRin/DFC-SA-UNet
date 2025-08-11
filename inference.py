import os
import time
import argparse
import yaml
import torch
import numpy as np
import matplotlib
import cv2
import csv
from PIL import Image, ImageFile
import torchvision.transforms as transforms
from tqdm import tqdm
import glob

# 設定非互動式後端以避免多線程問題
matplotlib.use('Agg')

# 允許載入被截斷的圖像檔案
ImageFile.LOAD_TRUNCATED_IMAGES = True

# 假設這些模組存在於您的專案結構中
# 由於無法直接執行，我們將模擬這些工廠和函式
from models.model_factory import ModelFactory
from utils.visualization import create_overlay, create_combined_visualization

# --- 核心程式碼 ---

def normalize_path(path):
    """
    標準化路徑以確保使用正確的斜線格式，以相容不同作業系統。
    """
    return path.replace('\\', '/')

def load_image(image_path, target_size=None):
    """
    載入並預處理圖像。
    如果 target_size 為 None，則以原始尺寸載入。
    支援標準圖片格式以及 .tif/.tiff 格式。
    """
    image_path = normalize_path(image_path)
    try:
        if image_path.lower().endswith(('.tif', '.tiff')):
            # 使用 OpenCV 讀取 TIFF 檔案
            image_np = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            if image_np is None:
                raise IOError(f"CV2 無法讀取圖像 {image_path}")
            # 處理可能的多通道 (例如 4通道 TIFF)
            if image_np.ndim == 3 and image_np.shape[2] == 4:
                image_np = cv2.cvtColor(image_np, cv2.COLOR_BGRA2BGR)
            original_image = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        else:
            # 使用 Pillow 讀取標準圖片
            original_image = Image.open(image_path).convert("RGB")
            original_image = np.array(original_image)
        
        image_for_tensor = Image.fromarray(original_image)
        
        if target_size:
            image_for_tensor = image_for_tensor.resize(target_size, Image.Resampling.BILINEAR)
        
        # 定義轉換流程
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        image_tensor = transform(image_for_tensor).unsqueeze(0)
        return image_tensor, original_image
    except Exception as e:
        print(f"錯誤: 無法載入圖像 {image_path}: {str(e)}")
        return None, None

def calculate_segmentation_metrics(pred_binary, gt_binary):
    """
    計算一組分割指標的原始計數（TP、FP、FN、TN）。
    """
    # 確保輸入是 0/1 的二進位格式
    pred_binary = (pred_binary > 0).astype(np.uint8)
    gt_binary = (gt_binary > 0).astype(np.uint8)

    pred_flat = pred_binary.flatten()
    gt_flat = gt_binary.flatten()

    # 計算 TP, FP, FN, TN
    tp = np.sum(pred_flat * gt_flat)
    fp = np.sum(pred_flat) - tp
    fn = np.sum(gt_flat) - tp
    tn = len(pred_flat) - (tp + fp + fn)
    
    # 返回原始計數
    return {'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn}

def predict_single_image(model, image_tensor, device):
    """
    對單個圖像張量進行直接預測。
    """
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        output = model(image_tensor)
        pred_prob = torch.sigmoid(output).squeeze(0).squeeze(0).cpu().numpy()
    return pred_prob

def predict_large_image(model, image, tile_size, overlap, device, use_tta=False):
    """
    使用滑動窗口對大圖進行預測，並可選擇性啟用測試時增強 (TTA)。
    """
    model.eval()
    h, w, _ = image.shape
    stride = tile_size - overlap
    
    prediction_canvas = np.zeros((h, w), dtype=np.float32)
    counts_canvas = np.zeros((h, w), dtype=np.float32)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    y_steps = range(0, h, stride)
    x_steps = range(0, w, stride)
    
    with torch.no_grad(), tqdm(total=len(y_steps) * len(x_steps), desc="   - 切割預測中", leave=False, unit="tile") as pbar:
        for y in y_steps:
            for x in x_steps:
                y_end = min(y + tile_size, h)
                x_end = min(x + tile_size, w)
                y_start = max(0, y_end - tile_size)
                x_start = max(0, x_end - tile_size)
                
                tile = image[y_start:y_end, x_start:x_end]
                tile_tensor = transform(Image.fromarray(tile)).unsqueeze(0).to(device)
                
                if use_tta:
                    pred_orig = torch.sigmoid(model(tile_tensor))
                    pred_hflip = torch.sigmoid(model(torch.flip(tile_tensor, [3])))
                    pred_hflip = torch.flip(pred_hflip, [3])
                    pred_vflip = torch.sigmoid(model(torch.flip(tile_tensor, [2])))
                    pred_vflip = torch.flip(pred_vflip, [2])
                    final_pred_tensor = (pred_orig + pred_hflip + pred_vflip) / 3.0
                    pred = final_pred_tensor.squeeze().cpu().numpy()
                else:
                    output = model(tile_tensor)
                    pred = torch.sigmoid(output).squeeze().cpu().numpy()
                
                prediction_canvas[y_start:y_end, x_start:x_end] += pred
                counts_canvas[y_start:y_end, x_start:x_end] += 1
                pbar.update(1)
                
    counts_canvas[counts_canvas == 0] = 1
    prediction_canvas /= counts_canvas
    
    return prediction_canvas

def save_prediction(original_image, pred_prob, pred_binary, output_dir, filename, gt_mask=None):
    """
    儲存所有視覺化結果，包括原始圖、熱圖、二值化預測、疊加圖以及對比圖。
    """
    output_dir = normalize_path(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    pred_binary_img = (pred_binary * 255).astype(np.uint8)

    gt_mask_for_vis = None
    if gt_mask is not None:
        gt_mask_for_vis = ((gt_mask > 0) * 255).astype(np.uint8)

    if gt_mask_for_vis is not None:
        combined_path = os.path.join(output_dir, f'{filename}_combined_view.png')
        create_combined_visualization(original_image, pred_binary_img, gt_mask_for_vis, filename, combined_path)
    
    individual_dir = os.path.join(output_dir, filename)
    os.makedirs(individual_dir, exist_ok=True)
    
    pred_heatmap = (pred_prob * 255).astype(np.uint8)
    pred_heatmap = cv2.applyColorMap(pred_heatmap, cv2.COLORMAP_JET)

    overlay = create_overlay(original_image, pred_binary)
    
    cv2.imwrite(os.path.join(individual_dir, 'original.png'), cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(individual_dir, 'pred_heatmap.png'), pred_heatmap)
    cv2.imwrite(os.path.join(individual_dir, 'pred_binary.png'), pred_binary_img)
    cv2.imwrite(os.path.join(individual_dir, 'pred_overlay.png'), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    
    if gt_mask_for_vis is not None:
        cv2.imwrite(os.path.join(individual_dir, 'ground_truth.png'), gt_mask_for_vis)

def main(args):
    """
    主推論函式
    """
    config_path = normalize_path(args.config)
    model_path = normalize_path(args.model)
    input_dir = normalize_path(args.input)
    output_dir = normalize_path(args.output)
    
    # --- 載入設定檔和模型 ---
    try:
        with open(config_path, 'r', encoding='utf8') as f:
            config = yaml.safe_load(f)
        print(f"成功從 {config_path} 載入設定檔")
    except Exception as e:
        print(f"載入設定檔錯誤: {str(e)}")
        raise
        
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.tta:
        print("測試時增強 (TTA) 已啟用。")
    if args.no_slide_window:
        print("模式: 單圖直接預測 (滑動窗口已停用)。")
        if args.resize:
            print(f"圖像將被縮放至: {args.resize[0]}x{args.resize[1]}")
    else:
        print("模式: 滑動窗口預測。")

    print(f"使用設備: {device}")
    
    try:
        if 'pretrained_path' in config['model']:
            config['model']['pretrained_path'] = None
        model = ModelFactory.get_model(config).to(device)
        checkpoint = torch.load(model_path, map_location=device)
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"從 checkpoint 載入模型權重 (epoch {checkpoint.get('epoch', 'N/A')})")
        else:
            model.load_state_dict(checkpoint)
            print("直接從 .pth 檔案載入模型權重")
        
        model.eval()
        print(f"模型 {config['model'].get('name', 'Unknown')} 從 {model_path} 載入成功")
    except Exception as e:
        print(f"載入模型錯誤: {str(e)}")
        raise
        
    os.makedirs(output_dir, exist_ok=True)
    
    # --- 檢查評估模式 ---
    evaluate_metrics = False
    original_img_dir = input_dir
    mask_dir = os.path.join(input_dir, 'mask')
    if os.path.isdir(mask_dir):
        original_img_dir_candidate = os.path.join(input_dir, 'original')
        if os.path.isdir(original_img_dir_candidate):
            original_img_dir = original_img_dir_candidate
            evaluate_metrics = True
            print("偵測到 'original' 和 'mask' 子目錄，將進行評估。")
        else:
            tqdm.write(f"警告: 找到 'mask' 目錄，但未找到 'original' 目錄。將不會進行評估。")

    image_files = []
    for ext in ['*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff']:
        image_files.extend(glob.glob(os.path.join(original_img_dir, ext)))

    if not image_files:
        print(f"在 {original_img_dir} 中找不到任何圖片檔案。")
        return

    all_metrics = []
    total_tp, total_fp, total_fn, total_tn = 0, 0, 0, 0
    
    # --- 主迴圈 ---
    with tqdm(image_files, desc="處理圖片中", unit="img") as progress_bar:
        for image_path in progress_bar:
            filename = os.path.splitext(os.path.basename(image_path))[0]
            progress_bar.set_description(f"處理中: {filename}")
            
            pred_prob = None
            if args.no_slide_window:
                # --- 直接預測模式 ---
                target_size = tuple(args.resize) if args.resize else None
                image_tensor, original_image = load_image(image_path, target_size=target_size)
                if image_tensor is None:
                    continue
                
                pred_prob_resized = predict_single_image(model, image_tensor, device)
                
                # 將預測結果縮放回原始圖像尺寸以進行評估和視覺化
                original_h, original_w = original_image.shape[:2]
                pred_prob = cv2.resize(pred_prob_resized, (original_w, original_h), interpolation=cv2.INTER_LINEAR)
            else:
                # --- 滑動窗口模式 ---
                _, original_image = load_image(image_path)
                if original_image is None:
                    continue
                pred_prob = predict_large_image(model, original_image, args.tile_size, args.overlap, device, use_tta=args.tta)
            
            pred_binary = (pred_prob > args.threshold).astype(np.uint8)
            
            gt_mask = None
            if evaluate_metrics:
                basename = os.path.splitext(os.path.basename(image_path))[0]
                # 尋找對應的 mask 檔案 (支援不同副檔名)
                mask_path = next(iter(glob.glob(os.path.join(mask_dir, f"{basename}.*"))), None)

                if mask_path and os.path.exists(mask_path):
                    _, gt_mask_full = load_image(mask_path)
                    if gt_mask_full is not None:
                        gt_mask = cv2.cvtColor(gt_mask_full, cv2.COLOR_RGB2GRAY)
                        
                        # --- 新增：將 GT 遮罩縮放到與模型輸出相同的尺寸 ---
                        pred_h, pred_w = pred_prob.shape
                        gt_mask_resized = cv2.resize(gt_mask, (pred_w, pred_h), interpolation=cv2.INTER_NEAREST)
                        gt_mask_resized = (gt_mask_resized > 128).astype(np.uint8)
                        # --- 修改結束 ---

                        gt_mask = (gt_mask > 128).astype(np.uint8)
                        
                        counts = calculate_segmentation_metrics(pred_binary, gt_mask_resized)
                        total_tp += counts['tp']
                        total_fp += counts['fp']
                        total_fn += counts['fn']
                        total_tn += counts['tn']

                        # 為了單獨儲存每張圖片的指標，我們在這裡計算一次
                        single_iou = counts['tp'] / (counts['tp'] + counts['fp'] + counts['fn'] + 1e-7)
                        single_dice = (2 * counts['tp']) / (2 * counts['tp'] + counts['fp'] + counts['fn'] + 1e-7)
                        single_acc = (counts['tp'] + counts['tn']) / (counts['tp'] + counts['tn'] + counts['fp'] + counts['fn'] + 1e-7)
                        single_recall = counts['tp'] / (counts['tp'] + counts['fn'] + 1e-7)
                        single_precision = counts['tp'] / (counts['tp'] + counts['fp'] + 1e-7)

                        metrics = {
                            'file': filename,
                            'iou': single_iou,
                            'dice_f1': single_dice,
                            'accuracy': single_acc,
                            'recall': single_recall,
                            'precision': single_precision,
                            'tp': counts['tp'],
                            'fp': counts['fp'],
                            'fn': counts['fn'],
                            'tn': counts['tn']
                        }
                        all_metrics.append(metrics)
                        
                        progress_bar.set_postfix(last_f1=f"{single_dice:.4f}")
                    else:
                        tqdm.write(f"警告: 無法載入對應的遮罩檔案 {mask_path}")
                else:
                    tqdm.write(f"警告: 找不到檔案 '{basename}' 對應的遮罩檔案")

            save_prediction(original_image, pred_prob, pred_binary, output_dir, filename, gt_mask=gt_mask)

    # --- 最終報告與 CSV 儲存 ---
    if evaluate_metrics and all_metrics:
        # --- 在控制台打印報告 ---
        # --- 計算全域指標 ---
        global_iou = total_tp / (total_tp + total_fp + total_fn + 1e-7)
        global_dice = (2 * total_tp) / (2 * total_tp + total_fp + total_fn + 1e-7)
        global_accuracy = (total_tp + total_tn) / (total_tp + total_tn + total_fp + total_fn + 1e-7)
        global_recall = total_tp / (total_tp + total_fn + 1e-7)
        global_precision = total_tp / (total_tp + total_fp + 1e-7)

        print("\n" + "="*80)
        print("--- 推論評估總結 ---")
        # --- 在控制台打印單個檔案的報告 ---
        if all_metrics:
            metric_keys = [k for k in all_metrics[0].keys() if k != 'file']
            header = f"{'File':<30}" + "".join([f"{key.upper():>12}" for key in metric_keys])
            print(header)
            print("-" * len(header))
            for m in all_metrics:
                row = f"{m['file']:<30}" + "".join([f"{m[key]:>12.4f}" for key in metric_keys])
                print(row)
        
        # --- 打印全域平均指標 ---
        print("\n" + "--- 全域平均指標 (Macro-Averaged) ---")
        print(f"{'指標':<15} | {'分數'}")
        print("-"*25)
        print(f"{'IoU':<15} | {global_iou:.4f}")
        print(f"{'Dice/F1':<15} | {global_dice:.4f}")
        print(f"{'Accuracy':<15} | {global_accuracy:.4f}")
        print(f"{'Recall':<15} | {global_recall:.4f}")
        print(f"{'Precision':<15} | {global_precision:.4f}")
        print("="*80)

        # --- 將結果儲存到 CSV ---
        # 決定 CSV 儲存路徑
        if args.csv_dir:
            csv_output_dir = normalize_path(args.csv_dir)
            os.makedirs(csv_output_dir, exist_ok=True)
            # 以模型設定檔的檔名來命名 CSV，避免覆蓋
            config_filename = os.path.splitext(os.path.basename(config_path))[0]
            csv_path = os.path.join(csv_output_dir, f'{config_filename}_metrics.csv')
        else:
            csv_path = os.path.join(output_dir, 'evaluation_metrics.csv')

        try:
            with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=['file'] + metric_keys)
                writer.writeheader()
                writer.writerows(all_metrics)
            print(f"\n評估指標已成功儲存至: {csv_path}")
        except IOError as e:
            print(f"\n錯誤: 無法寫入 CSV 檔案於 {csv_path}: {e}")

    print(f"\n推論完成。結果已儲存至 {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="使用訓練好的模型進行推論（支援大圖切割、TTA、多指標評估）")
    parser.add_argument("--config", type=str, required=True, help="模型設定檔路徑 (.yaml)")
    parser.add_argument("--model", type=str, required=True, help="模型權重路徑 (.pth)")
    parser.add_argument("--input", type=str, required=True, help="輸入圖像目錄。若需評分，此目錄下應有 'original' 和 'mask' 子目錄。")
    parser.add_argument("--output", type=str, default="results", help="輸出「圖片」結果的目錄")
    parser.add_argument("--csv_dir", type=str, default=None, help="(可選) 指定一個專門存放評估結果 .csv 檔案的目錄。如果未指定，CSV 將存放在 --output 目錄中。")
    parser.add_argument("--threshold", type=float, default=0.5, help="二值化預測的機率閾值")
    
    # 滑動窗口參數
    parser.add_argument("--tile_size", type=int, default=224, help="滑動窗口的圖塊大小 (僅在滑動窗口模式下有效)")
    parser.add_argument("--overlap", type=int, default=50, help="圖塊之間的重疊像素量 (僅在滑動窗口模式下有效)")
    
    # 新增的參數
    parser.add_argument("--resize", nargs=2, type=int, metavar=('WIDTH', 'HEIGHT'), help="將輸入圖像縮放至指定尺寸 (僅在 --no_slide_window 模式下建議使用)")
    parser.add_argument("--no_slide_window", action="store_true", help="停用滑動窗口，對整張圖 (可選縮放後) 直接預測")
    parser.add_argument("--tta", action="store_true", help="啟用測試時增強 (Test-Time Augmentation，僅在滑動窗口模式下有效)")
    
    args = parser.parse_args()
    
    main(args)