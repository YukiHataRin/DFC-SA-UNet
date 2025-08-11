[English](README.md)

# DFC-SA-UNet：用於跨領域語意分割的通用動態融合網路

本專案提供了一個基於 U-Net 架構，並融合了深度特徵融合 (Deep Feature Fusion) 和自注意力機制 (Self-Attention) 的影像分割模型，旨在提高醫學影像、衛星影像等分割任務的準確性。

## 專案特色

*   **混合架構**: 結合了 U-Net 的編碼器-解碼器結構與 Transformer 的自注意力機制，有效捕捉局部與全域特徵。
*   **靈活配置**: 所有訓練參數、模型選擇和資料路徑都透過 .yaml 檔案進行管理，方便實驗與調整。
*   **模組化設計**: 將模型、資料處理和訓練邏輯分離，易於擴充和維護。
*   **多模型支援**: 內建多種模型變體，包括純 U-Net、Transformer-UNet 以及多種消融實驗模型，方便進行效能比較。

## 下載資源

*   **資料集:** [點此從 Google Drive 下載](https://docs.google.com/uc?export=download&id=1E_nC-S4_Ofp-F3R_emhXp_i-4_aoAgS1)

    本研究產生和/或分析的資料集可應合理要求向通訊作者索取。
*   **訓練好的模型權重:** [點此從 Google Drive 下載](https://docs.google.com/uc?export=download&id=1your_pretrained_weights_id)

## 檔案結構

```
DFC-SA-UNet/
│
├── configs/                  # 存放所有模型和訓練的設定檔
│   ├── config_unet.yaml
│   ├── config_dfc-sa-res-block.yaml
│   └── ... (其他設定檔)
│
├── data/                     # (建議) 存放資料集
│   ├── train/
│   │   ├── original/
│   │   └── mask/
│   └── val/
│       ├── original/
│       └── mask/
│
├── models/                   # 存放所有模型架構的定義
│   ├── __init__.py
│   ├── data_loader.py
│   ├── model_factory.py
│   ├── transformer_unet.py
│   ├── unet_dfc_sa_ablation_attention.py
│   ├── unet_dfc_sa_ablation_branches.py
│   ├── unet_dfc_sa_ablation_fusion.py
│   ├── unet_dfc_sa_ablation_placement.py
│   ├── unet_dfc_sa_res.py
│   ├── unet.py
│   ├── vision_transformer.py
│   └── visualization.py
│
├── utils/                    # 存放輔助工具
│   ├── data_loader.py
│   ├── trainer.py
│   ├── metrics.py
│   └── visualization.py
│
├── train.py                  # 模型訓練腳本
├── inference.py              # 模型推論腳本
├── model_stats.py            # 計算模型參數與 FLOPs 的腳本
└── README.md                 # 專案說明文件
```

## 資料集架構

請依照以下建議的結構組織您的資料集。程式會自動在您於 config 檔案中指定的 `train_dir` 和 `val_dir` 路徑下，尋找 `original` 和 `mask` 這兩個子資料夾。

*   **訓練集:** `.../your_dataset/train/`
    *   **原始影像:** `.../train/original/`
    *   **標籤遮罩:** `.../train/mask/`
*   **驗證集:** `.../your_dataset/val/`
    *   **原始影像:** `.../val/original/`
    *   **標籤遮罩:** `.../val/mask/`

**注意:** `original` 資料夾中的原始影像和 `mask` 資料夾中其對應的遮罩檔案名稱必須相同（例如 `image1.png` 對應 `image1.png`）。

## 安裝

### 建立虛擬環境 (建議)

```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

### 安裝依賴套件

建議您建立一個 `requirements.txt` 檔案，並包含以下常用套件：

```
torch
torchvision
numpy
opencv-python
pyyaml
scikit-learn
matplotlib
tqdm
```

然後執行安裝：

```bash
pip install -r requirements.txt
```

## Config 配置說明

所有實驗的參數都在 `configs/` 資料夾中的 `.yaml` 檔案進行設定。以下是一個範例，說明了新的配置結構：

```yaml
training:
  num_epochs: 500
  batch_size: 4
  learning_rate: 0.01
  momentum: 0.9
  weight_decay: 0.0001
  num_workers: 2
  save_checkpoint_freq: 100
  device: 'auto'
  loss:
    type: 'bce_dice'
    params:
      bce_weight: 0.5
      dice_weight: 0.5

model:
  name: 'DFC-SA-Res-Block'
  in_channels: 3
  out_channels: 1
  features: [64, 128, 256, 512]
  pool_size: 4
  ablation_on_qk_channels: 8
  pretrained_path: null

dataset:
  train_dir: "path/to/your/dataset/train"
  val_dir: "path/to/your/dataset/val"
  img_size: [224, 224]
  augmentation: true

logging:
  log_dir: "path/to/your/logs"
  images_dir: "path/to/your/saved_images"
  save_best_worst_samples: 10
```

## 如何使用

### 1. 訓練模型

使用 `train.py` 腳本並透過 `--config` 參數指定要使用的設定檔。

```bash
python train.py --config configs/your_config_file.yaml
```

訓練過程中，腳本會：

1.  讀取指定的設定檔。
2.  建立模型和資料載入器。
3.  開始訓練，並在每個 epoch 結束後進行驗證。
4.  將驗證集上表現最好的模型權重儲存到 `logging.log_dir` 指定的路徑下。
5.  訓練日誌會顯示在終端機上。

### 2. 進行推論

使用 `inference.py` 腳本對新的影像進行分割。您需要指定設定檔、模型權重路徑以及儲存結果的路徑。

```bash
python inference.py \
    --config configs/your_config_file.yaml \
    --model_path path/to/your/best_model.pth \
    --output_dir results/
```

*   `--config`: 訓練時使用的設定檔，以確保模型結構一致。
*   `--model_path`: 已訓練好的模型權重檔案 (`.pth`)。
*   `--output_dir`: 儲存分割結果遮罩圖的路徑。

### 3. 計算模型統計數據

若要評估模型的複雜度，可以使用 `model_stats.py` 來計算其參數數量和浮點運算次數 (FLOPs)。

```bash
python model_stats.py --config configs/your_config_file.yaml
```

此腳本會根據設定檔建立模型，並輸出以下資訊：

*   總參數數量 (Total parameters)
*   可訓練參數數量 (Trainable parameters)
*   模型的 GFLOPs (Giga Floating Point Operations per Second)

```