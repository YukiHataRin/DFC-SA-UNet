import os
import sys
import torch
import yaml
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from prettytable import PrettyTable
from thop import profile, clever_format
from ptflops import get_model_complexity_info
from models.model_factory import ModelFactory

def count_parameters(model):
    """Calculate model parameters"""
    table = PrettyTable(["Module", "Parameters"])
    total_params = 0
    trainable_params = 0
    non_trainable_params = 0
    
    for name, parameter in model.named_parameters():
        param = parameter.numel()
        table.add_row([name, param])
        total_params += param
        if parameter.requires_grad:
            trainable_params += param
        else:
            non_trainable_params += param
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'non_trainable': non_trainable_params,
        'table': table
    }

def get_model_size(model):
    """計算模型大小（MB）"""
    torch.save(model.state_dict(), "temp.pth")
    size = os.path.getsize("temp.pth") / (1024 * 1024.0)
    os.remove("temp.pth")
    return size

def plot_parameter_distribution(model, output_dir, model_name):
    """Plot parameter distribution pie chart"""
    module_params = {}
    total_params = 0
    
    for name, parameter in model.named_parameters():
        module_name = name.split('.')[0]
        param_count = parameter.numel()
        module_params[module_name] = module_params.get(module_name, 0) + param_count
        total_params += param_count
    
    # Calculate percentages
    module_percentages = {k: (v/total_params)*100 for k, v in module_params.items()}
    
    # Create pie chart
    plt.figure(figsize=(10, 8))
    plt.pie(module_percentages.values(), labels=module_percentages.keys(), autopct='%1.1f%%')
    plt.title(f'Parameter Distribution - {model_name}')
    
    # Save to model directory
    model_dir = os.path.join(output_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)
    plt.savefig(os.path.join(model_dir, 'parameter_distribution.png'))
    plt.close()

def generate_model_summary(model, input_size):
    """生成模型架構摘要"""
    def register_hook(module):
        def hook(module, input, output):
            class_name = str(module.__class__.__name__)
            module_idx = len(summary)
            
            m_key = f"{class_name}-{module_idx+1}"
            summary[m_key] = {}
            summary[m_key]["input_shape"] = list(input[0].size())
            
            # 處理不同類型的輸出
            if isinstance(output, tuple):
                # 如果是元組，使用第一個元素（通常是主要輸出）
                summary[m_key]["output_shape"] = list(output[0].size())
            else:
                # 如果是張量，直接使用
                summary[m_key]["output_shape"] = list(output.size())
            
            params = 0
            for p in module.parameters():
                params += p.numel()
            summary[m_key]["params"] = params
        
        if not isinstance(module, torch.nn.Sequential) and \
           not isinstance(module, torch.nn.ModuleList) and \
           not (module == model):
            hooks.append(module.register_forward_hook(hook))
    
    # 初始化
    summary = {}
    hooks = []
    
    # 註冊鉤子
    model.apply(register_hook)
    
    # 進行一次前向傳播
    x = torch.zeros(input_size).to(next(model.parameters()).device)
    model(x)
    
    # 移除鉤子
    for h in hooks:
        h.remove()
    
    return summary

def save_stats_report(stats, output_dir, model_name):
    """Save statistics report"""
    # Create model directory
    model_dir = os.path.join(output_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)
    
    # Save as CSV
    df = pd.DataFrame(stats)
    df.to_csv(os.path.join(model_dir, 'model_stats.csv'))
    
    # Save as text report
    with open(os.path.join(model_dir, 'model_stats.txt'), 'w', encoding='utf-8') as f:
        f.write(f"Model Statistics Report - {model_name}\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("Parameter Statistics:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Total Parameters: {stats['total_params']:,}\n")
        f.write(f"Trainable Parameters: {stats['trainable_params']:,}\n")
        f.write(f"Non-trainable Parameters: {stats['non_trainable_params']:,}\n\n")
        
        f.write("Model Size:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Size: {stats['model_size']:.2f} MB\n\n")
        
        f.write("Computational Complexity:\n")
        f.write("-" * 30 + "\n")
        f.write(f"FLOPs: {stats['flops']}\n")
        f.write(f"MACs: {stats['macs']}\n")

def main(config_path, output_dir, input_size):
    """Main function"""
    # Load configuration
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Get model name and instance
    model_name = config['model']['name']
    model = ModelFactory.get_model(config)
    model.eval()
    
    # Calculate parameter statistics
    param_stats = count_parameters(model)
    
    # Calculate model size
    model_size = get_model_size(model)
    
    # Calculate computational complexity
    macs, params = get_model_complexity_info(model, input_size[1:], as_strings=True,
                                           print_per_layer_stat=False, verbose=False)
    
    # Generate model summary
    model_summary = generate_model_summary(model, input_size)
    
    # Plot parameter distribution
    plot_parameter_distribution(model, output_dir, model_name)
    
    # Organize statistics
    stats = {
        'total_params': param_stats['total'],
        'trainable_params': param_stats['trainable'],
        'non_trainable_params': param_stats['non_trainable'],
        'model_size': model_size,
        'flops': macs,
        'macs': params,
        'model_summary': model_summary
    }
    
    # Save report
    save_stats_report(stats, output_dir, model_name)
    
    # Print summary
    print(f"\nModel Statistics Summary - {model_name}:")
    print("=" * 50)
    print(f"Total Parameters: {stats['total_params']:,}")
    print(f"Trainable Parameters: {stats['trainable_params']:,}")
    print(f"Non-trainable Parameters: {stats['non_trainable_params']:,}")
    print(f"Model Size: {stats['model_size']:.2f} MB")
    print(f"Computational Complexity:")
    print(f"  - MACs: {stats['macs']}")
    print(f"  - FLOPs: {stats['flops']}")
    print(f"\nDetailed report saved to: {os.path.join(output_dir, model_name)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='模型統計分析工具')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                      help='配置文件路徑')
    parser.add_argument('--output', type=str, default='model_stats',
                      help='輸出目錄')
    parser.add_argument('--batch_size', type=int, default=1,
                      help='批次大小')
    parser.add_argument('--height', type=int, help='輸入高度')
    parser.add_argument('--width', type=int, help='輸入寬度')
    parser.add_argument('--channels', type=int, help='輸入通道數')
    
    args = parser.parse_args()
    
    # 從配置文件讀取輸入尺寸
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 使用命令行參數或配置文件中的值
    height = args.height or config['dataset']['img_size'][0]
    width = args.width or config['dataset']['img_size'][1]
    channels = args.channels or config['model']['in_channels']
    
    input_size = (args.batch_size, channels, height, width)
    
    main(args.config, args.output, input_size)