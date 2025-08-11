[繁體中文](README.zh-TW.md)

# DFC-SA-UNet: A Generalizable Dynamic Fusion Network for Cross-Domain Semantic Segmentation

This project provides an image segmentation model based on the U-Net architecture, incorporating Deep Feature Fusion and Self-Attention mechanisms, aimed at improving the accuracy of segmentation tasks for medical and satellite imagery.

## Features

*   **Hybrid Architecture**: Combines the encoder-decoder structure of U-Net with the self-attention mechanism of Transformers to effectively capture both local and global features.
*   **Flexible Configuration**: All training parameters, model selection, and data paths are managed through `.yaml` files for easy experimentation and tuning.
*   **Modular Design**: Separates model, data processing, and training logic for easy extension and maintenance.
*   **Multi-Model Support**: Includes built-in support for multiple model variations, including a pure U-Net, Transformer-UNet, and various ablation study models for performance comparison.

## Download Resources

*   **Datasets:** [Download from Google Drive](https://docs.google.com/uc?export=download&id=1E_nC-S4_Ofp-F3R_emhXp_i-4_aoAgS1)

    The datasets generated and/or analyzed during the current study are available from the corresponding author on reasonable request.
*   **Trained Model Weights:** [Download from Google Drive](https://docs.google.com/uc?export=download&id=1your_pretrained_weights_id)

## File Structure

```
DFC-SA-UNet/
│
├── configs/                  # Stores all configuration files for models and training
│   ├── config_unet.yaml
│   ├── config_dfc-sa-res-block.yaml
│   └── ... (other configuration files)
│
├── data/                     # (Recommended) Stores the dataset
│   ├── train/
│   │   ├── original/
│   │   └── mask/
│   └── val/
│       ├── original/
│       └── mask/
│
├── models/                   # Stores all model architecture definitions
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
├── utils/                    # Stores utility scripts
│   ├── data_loader.py
│   ├── trainer.py
│   ├── metrics.py
│   └── visualization.py
│
├── train.py                  # Model training script
├── inference.py              # Model inference script
├── model_stats.py            # Script to calculate model parameters and FLOPs
└── README.md                 # Project documentation
```

## Dataset Structure

Please organize your dataset according to the following recommended structure. The program will automatically look for `original` and `mask` subfolders under the `train_dir` and `val_dir` paths specified in your config file.

*   **Training Set:** `.../your_dataset/train/`
    *   **Original Images:** `.../train/original/`
    *   **Masks:** `.../train/mask/`
*   **Validation Set:** `.../your_dataset/val/`
    *   **Original Images:** `.../val/original/`
    *   **Masks:** `.../val/mask/`

**Note:** The filenames of the original images in the `original` folder and their corresponding masks in the `mask` folder must be identical (e.g., `image1.png` corresponds to `image1.png`).

## Installation

### Create a Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

### Install Dependencies

It is recommended to create a `requirements.txt` file and include the following common packages:

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

Then run the installation:

```bash
pip install -r requirements.txt
```

## Config Configuration

All experiment parameters are set in the `.yaml` files in the `configs/` folder. The following is an example illustrating the new configuration structure:

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

## How to Use

### 1. Train the Model

Use the `train.py` script and specify the configuration file to use with the `--config` parameter.

```bash
python train.py --config configs/your_config_file.yaml
```

During training, the script will:

1.  Read the specified configuration file.
2.  Create the model and data loaders.
3.  Start training and perform validation after each epoch.
4.  Save the best performing model weights on the validation set to the path specified by `logging.log_dir`.
5.  Training logs will be displayed in the terminal.

### 2. Perform Inference

Use the `inference.py` script to segment new images. You need to specify the configuration file, model weights path, and the path to save the results.

```bash
python inference.py \
    --config configs/your_config_file.yaml \
    --model_path path/to/your/best_model.pth \
    --output_dir results/
```

*   `--config`: The configuration file used during training to ensure model consistency.
*   `--model_path`: The trained model weights file (`.pth`).
*   `--output_dir`: The path to save the resulting segmentation masks.

### 3. Calculate Model Statistics

To evaluate the complexity of the model, you can use `model_stats.py` to calculate its number of parameters and Floating Point Operations (FLOPs).

```bash
python model_stats.py --config configs/your_config_file.yaml
```

This script will create the model based on the configuration file and output the following information:

*   Total parameters
*   Trainable parameters
*   GFLOPs (Giga Floating Point Operations per Second) of the model

```