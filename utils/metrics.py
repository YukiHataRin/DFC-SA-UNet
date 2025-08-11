import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn

def dice_loss(pred, target, smooth=1.0):
    """
    計算 Dice Loss
    
    Args:
        pred: 預測結果 (經過 sigmoid 的輸出)
        target: 真實標籤
        smooth: 平滑因子，避免分母為零
        
    Returns:
        dice_loss: 1 - dice係數
    """
    # 將輸入攤平為一維
    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)
    
    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    return 1 - dice

def tversky_loss(pred, target, alpha=0.5, beta=0.5, smooth=1.0):
    """
    計算 Tversky Loss，是 Dice Loss 的一般化形式
    
    Args:
        pred: 預測結果 (經過 sigmoid 的輸出)
        target: 真實標籤
        alpha: 假陽性的權重
        beta: 假陰性的權重
        smooth: 平滑因子，避免分母為零
        
    Returns:
        tversky_loss: 1 - tversky係數
    """
    # 將輸入攤平為一維
    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)
    
    # 計算真陽性、假陽性和假陰性
    tp = (pred * target).sum()
    fp = ((1-target) * pred).sum()
    fn = (target * (1-pred)).sum()
    
    tversky = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
    return 1 - tversky

class BCEDiceLoss(nn.Module):
    """
    結合 BCE Loss 和 Dice Loss 的混合損失函數
    """
    def __init__(self, weight_bce=1.0, weight_dice=1.0):
        super(BCEDiceLoss, self).__init__()
        self.weight_bce = weight_bce
        self.weight_dice = weight_dice
        self.bce = nn.BCELoss()
    
    def forward(self, inputs, targets, smooth=1.0):
        """
        計算 BCE + Dice Loss
        
        Args:
            inputs: 模型預測（應介於 0~1 之間，如經過 Sigmoid）
            targets: 真實標籤
            smooth: Dice Loss 的平滑因子
            
        Returns:
            weighted_loss: 加權後的 BCE + Dice Loss
        """
        bce_loss = self.bce(inputs, targets)
        dice_l = dice_loss(inputs, targets, smooth)
        
        # 加權組合兩種損失
        return self.weight_bce * bce_loss + self.weight_dice * dice_l

class DiceLoss(nn.Module):
    """
    Dice 損失函數。
    """
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        preds = torch.sigmoid(logits)
        preds = preds.contiguous().view(-1)
        targets = targets.contiguous().view(-1)
        intersection = (preds * targets).sum()
        dice_score = (2. * intersection + self.smooth) / (preds.sum() + targets.sum() + self.smooth)
        return 1 - dice_score

class JointLoss(nn.Module):
    """
    修正後的聯合損失函數，接收經過 sigmoid 的輸出。
    """
    def __init__(self, bce_weight=1.0, dice_weight=1.0, contour_weight=1.0):
        super(JointLoss, self).__init__()
        self.bce_loss = nn.BCELoss()  # 改為 BCELoss，因為輸入已經是機率
        self.dice_loss = DiceLoss()
        
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.contour_weight = contour_weight
        
        kernel = torch.tensor([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('contour_kernel', kernel)

    def forward(self, sigmoid_outputs, targets):
        # 檢查輸入是否包含NaN或無窮大值
        if torch.isnan(sigmoid_outputs).any() or torch.isinf(sigmoid_outputs).any():
            print("Warning: NaN or Inf detected in sigmoid outputs")
            sigmoid_outputs = torch.nan_to_num(sigmoid_outputs, nan=0.5, posinf=1.0, neginf=0.0)
        
        if torch.isnan(targets).any() or torch.isinf(targets).any():
            print("Warning: NaN or Inf detected in targets")
            targets = torch.nan_to_num(targets, nan=0.0)
        
        # 確保 sigmoid_outputs 在有效範圍內 [0, 1]
        sigmoid_outputs = torch.clamp(sigmoid_outputs, min=1e-7, max=1-1e-7)
        
        # 1. 計算分割損失 (L_SEG)
        # BCE loss 直接使用 sigmoid 輸出
        l_bce = self.bce_loss(sigmoid_outputs, targets)
        
        # Dice loss 需要將 sigmoid 輸出轉換為 logits
        # 使用數值穩定的 logit 轉換
        pred_logits = torch.log(sigmoid_outputs / (1 - sigmoid_outputs))
        l_dice = self.dice_loss(pred_logits, targets)
        
        # 檢查分割損失是否為NaN
        if torch.isnan(l_bce).any():
            print("Warning: NaN detected in BCE loss")
            l_bce = torch.tensor(0.0, device=sigmoid_outputs.device, requires_grad=True)
        if torch.isnan(l_dice).any():
            print("Warning: NaN detected in Dice loss")
            l_dice = torch.tensor(0.0, device=sigmoid_outputs.device, requires_grad=True)
            
        l_seg = (self.bce_weight * l_bce) + (self.dice_weight * l_dice)
        
        # 2. 計算輪廓懲罰損失 (L_CP)
        
        # 確保 contour_kernel 在正確的設備上
        device = sigmoid_outputs.device
        contour_kernel = self.contour_kernel.to(device)
        
        # 對 sigmoid 輸出進行輪廓檢測
        pred_contour = F.conv2d(sigmoid_outputs, contour_kernel, padding=1)
        
        # 對 target 計算輪廓
        target_contour = F.conv2d(targets, contour_kernel, padding=1)
        target_contour = torch.clamp(target_contour, 0, 1).detach()
        
        # 將輪廓預測結果限制在合理範圍內
        pred_contour = torch.clamp(pred_contour, min=0.0, max=1.0)
        
        # 使用 BCE 損失計算輪廓損失
        l_cp = self.bce_loss(pred_contour, target_contour)
        
        # 檢查輪廓損失是否為NaN
        if torch.isnan(l_cp).any():
            print("Warning: NaN detected in contour loss")
            l_cp = torch.tensor(0.0, device=sigmoid_outputs.device, requires_grad=True)
        
        # 3. 計算最終的聯合損失
        total_loss = l_seg + (self.contour_weight * l_cp)
        
        # 最終檢查
        if torch.isnan(total_loss).any():
            print("Warning: NaN detected in total loss, returning fallback loss")
            total_loss = l_bce + l_dice  # 使用簡單的BCE+Dice作為後備
        
        return total_loss

def dice_coefficient(pred, target, smooth=1):
    """
    計算Dice係數
    
    參數:
        pred: 預測值（已經過sigmoid）
        target: 真實標籤
        smooth: 平滑項，防止分母為0
    """
    pred = pred.contiguous()
    target = target.contiguous()
    
    intersection = (pred * target).sum(dim=2).sum(dim=2)
    dice = (2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)
    return dice.mean()

def iou_score(pred, target, smooth=1):
    """
    計算IoU分數
    
    參數:
        pred: 預測值（已經過sigmoid）
        target: 真實標籤
        smooth: 平滑項，防止分母為0
    """
    pred = pred.contiguous()
    target = target.contiguous()
    
    intersection = (pred * target).sum(dim=2).sum(dim=2)
    union = pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou.mean()

def calculate_metrics(pred, target, loss_type='dice', loss_params=None):
    """
    計算各種評估指標
    
    Args:
        pred: 預測結果 (經過 sigmoid 的輸出)
        target: 真實標籤
        loss_type: 損失函數類型，可選 'dice', 'tversky', 'bce_dice', 'joint'
        loss_params: 損失函數參數
        
    Returns:
        metrics: 包含各種評估指標的字典
    """
    if loss_params is None:
        loss_params = {}
    
    # 二值化預測結果
    pred_binary = (pred > 0.5).float()
    
    # 計算 IoU
    intersection = (pred_binary * target).sum().item()
    union = (pred_binary + target).sum().item() - intersection
    iou = intersection / (union + 1e-7)
    
    # 計算 Dice 係數
    dice_coef = (2. * intersection) / (pred_binary.sum().item() + target.sum().item() + 1e-7)
    
    # 計算損失
    if loss_type == 'dice':
        loss = dice_loss(pred, target)
    elif loss_type == 'tversky':
        alpha = loss_params.get('alpha', 0.5)
        beta = loss_params.get('beta', 0.5)
        loss = tversky_loss(pred, target, alpha, beta)
    elif loss_type == 'bce_dice':
        weight_bce = loss_params.get('weight_bce', 1.0)
        weight_dice = loss_params.get('weight_dice', 1.0)
        bce_dice_loss = BCEDiceLoss(weight_bce, weight_dice)
        loss = bce_dice_loss(pred, target)
    elif loss_type == 'joint':
        bce_weight = loss_params.get('bce_weight', 1.0)
        dice_weight = loss_params.get('dice_weight', 1.0)
        contour_weight = loss_params.get('contour_weight', 1.0)
        joint_loss = JointLoss(bce_weight, dice_weight, contour_weight)
        # JointLoss 現在直接接收 sigmoid 輸出
        loss = joint_loss(pred, target)
    else:
        raise ValueError(f"不支持的損失函數類型: {loss_type}")
    
    return {
        'loss': loss,
        'iou': iou,
        'dice': dice_coef
    } 