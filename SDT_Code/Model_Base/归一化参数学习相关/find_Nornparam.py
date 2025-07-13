import torch
import matplotlib.pyplot as plt
import numpy as np
from collections import OrderedDict

def visualize_learnable_norms(checkpoint_path, save_path=None):
    # 加载checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint.get('model', checkpoint.get('state_dict', checkpoint))
    
    # 提取所有learnable_norm参数
    norms = OrderedDict()
    for name, param in state_dict.items():
        if 'learnable_norm' in name:
            norms[name] = param.item()
    
    if not norms:
        print("未找到learnable_norm参数")
        return

    # 创建可视化
    plt.figure(figsize=(12, 6))
    
    # 条形图
    plt.subplot(1, 2, 1)
    names = list(norms.keys())
    values = list(norms.values())
    colors = plt.cm.viridis(np.linspace(0, 1, len(values)))
    
    bars = plt.bar(range(len(values)), values, color=colors)
    plt.xticks(range(len(values)), [f'L{i}' for i in range(len(values))], rotation=45)
    plt.ylabel('Parameter Value')
    plt.title('Learnable Norm Values')
    
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
    plt.title('Value Distribution')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"可视化结果已保存至: {save_path}")
    plt.show()

    # 打印统计信息
    print("\n参数统计:")
    print(f"平均值: {np.mean(values):.4f}")
    print(f"标准差: {np.std(values):.4f}")
    print(f"最大值: {max(values):.4f} (位于 {names[values.index(max(values))]})")
    print(f"最小值: {min(values):.4f} (位于 {names[values.index(min(values))]})")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', type=str, help='Path to checkpoint file')
    parser.add_argument('--save', type=str, default=None, help='Save path for figure')
    args = parser.parse_args()
    visualize_learnable_norms(args.checkpoint, args.save)