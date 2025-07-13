import torch
import matplotlib.pyplot as plt
import numpy as np
from collections import OrderedDict

def extract_and_plot_max_values(checkpoint_path, save_path=None):
    # 加载checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint.get('model', checkpoint.get('state_dict', checkpoint))
    
    # 提取所有max_value参数
    max_values = OrderedDict()
    for name, param in state_dict.items():
        if 'max_value' in name and param.numel() == 1:  # 确保是标量参数
            max_values[name] = param.item()
    
    if not max_values:
        print("未找到max_value参数")
        return

    # 创建可视化
    plt.figure(figsize=(12, 6))
    
    # 条形图
    plt.subplot(1, 2, 1)
    names = [f'L{i}' for i in range(len(max_values))]
    values = list(max_values.values())
    colors = plt.cm.plasma(np.linspace(0, 1, len(values)))
    
    bars = plt.bar(names, values, color=colors)
    plt.xticks(rotation=45)
    plt.ylabel('Max Value')
    plt.title('Learned Max Values')
    
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.2f}', ha='center', va='bottom')
    
    # 分布直方图
    plt.subplot(1, 2, 2)
    plt.hist(values, bins=15, alpha=0.7, color='skyblue', edgecolor='black')
    plt.xlabel('Value Range')
    plt.ylabel('Frequency')
    plt.title('Max Value Distribution')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"可视化结果已保存至: {save_path}")
    plt.show()

    # 打印统计信息
    print("\n参数统计:")
    print(f"参数总数: {len(max_values)}")
    print(f"平均值: {np.mean(values):.4f}")
    print(f"标准差: {np.std(values):.4f}")
    print(f"最大值: {max(values):.4f}")
    print(f"最小值: {min(values):.4f}")

    # 打印所有max_value参数
    print("\n所有max_value参数:")
    for name, value in max_values.items():
        print(f"{name}: {value:.6f}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', type=str, help='Path to checkpoint file')
    parser.add_argument('--save', type=str, default=None, help='Save path for figure')
    args = parser.parse_args()
    extract_and_plot_max_values(args.checkpoint, args.save)