import torch
import torch.nn as nn
from .unet_dfc_sa_res import UNetDFCSARes
from .transformer_unet import TransUNet, get_r50_b16_config
from .unet import UNet
from .vision_transformer import VisionTransformerForSegmentation
# 新增消融實驗模型
from .unet_dfc_sa_ablation_branches import UNet_Baseline, UNet_AttentionOnly
from .unet_dfc_sa_ablation_fusion import UNet_AdditionFusion, UNet_ConcatFusion
from .unet_dfc_sa_ablation_attention import UNet_FullResAttention
from .unet_dfc_sa_ablation_placement import UNet_EncoderOnlyDFC, UNet_DecoderOnlyDFC, UNet_BothStandardConv
import math

class ModelFactory:
    """
    模型工廠類，用於根據配置創建各種模型實例
    
    兩種使用方式:
    1. ModelFactory.get_model(config) - 靜態方法
    2. factory = ModelFactory(config); model = factory.create_model() - 實例方法
    """
    
    def __init__(self, config=None):
        """
        初始化模型工廠
        
        參數:
            config: 配置字典，可選
        """
        self.config = config
    
    def create_model(self, config=None):
        """
        實例方法：根據配置創建模型
        
        參數:
            config: 配置字典，如果在初始化時已提供則可省略
            
        返回:
            創建的模型實例
        """
        # 使用傳入的配置或初始化時的配置
        if config is None:
            if self.config is None:
                raise ValueError("必須提供配置")
            config = self.config
        
        return ModelFactory._create_model_impl(config)
    
    @staticmethod
    def get_model(config):
        """
        靜態方法：根據配置創建模型
        
        參數:
            config: 配置字典
            
        返回:
            創建的模型實例
        """
        # 創建模型
        model = ModelFactory._create_model_impl(config)
        
        # 如果有預訓練權重，則載入
        if config['model'].get('pretrained_path'):
            try:
                model.load_state_dict(torch.load(config['model']['pretrained_path'], weights_only=False))
                print(f"成功載入預訓練權重: {config['model']['pretrained_path']}")
            except Exception as e:
                print(f"載入預訓練權重失敗: {e}")
        
        return model
    
    @staticmethod
    def _create_model_impl(config):
        """
        內部實現：創建模型的具體邏輯
        
        參數:
            config: 配置字典
            
        返回:
            創建的模型實例（未載入預訓練權重）
        """
        # 從配置中獲取模型參數
        model_name = config['model']['name']
        in_channels = config['model'].get('in_channels', 3)
        out_channels = config['model'].get('out_channels', 1)
        features = config['model'].get('features', [64, 128, 256, 512])
        pool_size = config['model'].get('pool_size', 8)
        ablation_on_qk_channels = config['model'].get('ablation_on_qk_channels', 8)
        
        # U-Net系列模型
        if model_name == 'UNet':
            bilinear = config['model'].get('bilinear', False)
            return UNet(
                n_channels=in_channels,
                n_classes=out_channels,
                bilinear=bilinear
            )
        
        # DFC-SA系列模型
        elif model_name == 'DFC-SA-Res-Block':
            return UNetDFCSARes(
                in_channels=in_channels,
                out_channels=out_channels,
                features=features,
                pool_size=pool_size,
                ablation_on_qk_channels=ablation_on_qk_channels
            )
        
        # 其他基礎模型
        elif model_name in ['TransformerUNet', 'TransUNet']:
            # --- 這是為官方版 TransUNet 新增的邏輯 ---
            print("正在創建官方版 TransUNet (R50-ViT-B_16)...")

            # 1. 獲取 R50-ViT-B_16 的基礎設定
            vit_config = get_r50_b16_config()

            # 2. 從您的 YAML 配置中讀取並覆蓋關鍵參數
            img_size_config = config['dataset'].get('img_size', [224, 224])
            img_size = img_size_config[0] if isinstance(img_size_config, list) else img_size_config
            
            vit_config.n_classes = out_channels
            
            # 官方版 TransUNet 對輸入通道數沒有直接的參數，它預設為3
            # 如果您的 in_channels 不是3，模型內部會自動處理單通道輸入
            if in_channels != 3:
                print(f"注意：官方版 TransUNet 預設處理3通道輸入。您的 in_channels={in_channels}，模型會將單通道複製為3通道。")

            # 動態計算 patch grid size
            # 官方版 ViT-B_16 的 patch size 是 16
            vit_patch_size = 16 
            vit_config.patches.grid = (img_size // vit_patch_size, img_size // vit_patch_size)

            # 3. 實例化模型
            return TransUNet(config=vit_config, img_size=img_size, num_classes=out_channels)
        
        # Vision Transformer系列模型
        elif model_name == 'VisionTransformerSegmentation': # Or whatever name you choose
            img_dim = config['model'].get('img_dim', 224)
            patch_dim = config['model'].get('patch_dim', 16)
            # ... get other ViT specific params from config ...
            segmentation_head_upsample_layers = config['model'].get('segmentation_head_upsample_layers', int(math.log2(patch_dim)) if patch_dim > 0 and (patch_dim & (patch_dim - 1) == 0) else 4) # Default to 4 if not a power of 2 or not specified

            return VisionTransformerForSegmentation(
                img_dim=img_dim,
                patch_dim=patch_dim,
                in_channels=in_channels,
                num_classes=out_channels,
                embed_dim=config['model'].get('embed_dim', 768),
                num_layers=config['model'].get('num_layers', 12),
                num_heads=config['model'].get('num_heads', 12),
                mlp_dim=config['model'].get('mlp_dim', 3072),
                dropout=config['model'].get('dropout', 0.1),
                segmentation_head_upsample_layers=segmentation_head_upsample_layers
            )

        
        # --- 消融實驗模型 ---
        # 實驗一：核心模塊的雙分支有效性
        elif model_name == 'UNet_Baseline':
            return UNet_Baseline(in_channels, out_channels, features)
        elif model_name == 'UNet_AttentionOnly':
            return UNet_AttentionOnly(in_channels, out_channels, features, pool_size)
        
        # 實驗二：動態融合機制的有效性
        elif model_name == 'UNet_AdditionFusion':
            return UNet_AdditionFusion(in_channels, out_channels, features, pool_size)
        elif model_name == 'UNet_ConcatFusion':
            return UNet_ConcatFusion(in_channels, out_channels, features, pool_size)
        
        # 實驗三：輕量級自注意力的效率
        elif model_name == 'UNet_FullResAttention':
            return UNet_FullResAttention(in_channels, out_channels, features)
        
        # 實驗四：DFC-SA Block 的放置位置
        elif model_name == 'UNet_EncoderOnlyDFC':
            return UNet_EncoderOnlyDFC(in_channels, out_channels, features, pool_size)
        elif model_name == 'UNet_DecoderOnlyDFC':
            return UNet_DecoderOnlyDFC(in_channels, out_channels, features, pool_size)
        elif model_name == 'UNet_BothStandardConv':
            return UNet_BothStandardConv(in_channels, out_channels, features)
        
        else:
            raise ValueError(f'不支援的模型類型: {model_name}')
