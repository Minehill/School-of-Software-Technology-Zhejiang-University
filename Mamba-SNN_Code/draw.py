import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse

def plot_training_log(log_file_path):
    """
    读取并解析log.txt文件，然后绘制训练过程中的准确率和损失曲线。
    改进功能：
    1. 通过命令行参数指定日志文件路径
    2. 在图中标出test_acc1最大的地方，标注其大小和epoch
    """
    # 检查文件是否存在
    if not os.path.exists(log_file_path):
        print(f"错误: 找不到日志文件 '{log_file_path}'")
        print("请检查路径是否正确，或者训练是否已经产生了日志文件。")
        return

    # 读取日志文件
    log_data = []
    try:
        with open(log_file_path, 'r') as f:
            for line in f:
                log_data.append(json.loads(line))
    except Exception as e:
        print(f"读取或解析文件时出错: {e}")
        return

    if not log_data:
        print("日志文件为空，无法绘图。")
        return

    # 将数据转换为Pandas DataFrame，方便处理
    df = pd.DataFrame(log_data)

    # 确保关键列存在
    required_cols = ['epoch', 'test_acc1', 'test_acc5', 'train_loss', 'test_loss']
    if not all(col in df.columns for col in required_cols):
        print("日志文件中缺少必要的列（如 'test_acc1', 'test_acc5'等）。")
        print(f"可用列: {df.columns.tolist()}")
        return

    # 找到test_acc1的最大值及其epoch
    max_acc1_idx = df['test_acc1'].idxmax()
    max_acc1 = df.loc[max_acc1_idx, 'test_acc1']
    max_acc1_epoch = df.loc[max_acc1_idx, 'epoch']
    
    # 找到test_acc5的最大值及其epoch
    max_acc5_idx = df['test_acc5'].idxmax()
    max_acc5 = df.loc[max_acc5_idx, 'test_acc5']
    max_acc5_epoch = df.loc[max_acc5_idx, 'epoch']

    # 设置绘图风格
    sns.set_theme(style="whitegrid")

    # 创建一个2x1的子图布局
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    fig.suptitle('Training Performance Analysis', fontsize=16)

    # --- 绘制准确率曲线 (ax1) ---
    ax1.plot(df['epoch'], df['test_acc1'], 'o-', label='Validation Top-1 Acc', color='C0')
    ax1.plot(df['epoch'], df['test_acc5'], 's--', label='Validation Top-5 Acc', color='C1')
    
    # 标记Top-1最大准确率点
    ax1.plot(max_acc1_epoch, max_acc1, 'ro', markersize=10, 
             label=f'Max Top-1 Acc: {max_acc1:.2f}% @ Epoch {max_acc1_epoch}')
    
    # 标记Top-5最大准确率点
    ax1.plot(max_acc5_epoch, max_acc5, 'gs', markersize=10, 
             label=f'Max Top-5 Acc: {max_acc5:.2f}% @ Epoch {max_acc5_epoch}')
    
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('Validation Accuracy (Top-1 & Top-5)')
    ax1.legend(loc='lower right')
    ax1.grid(True, which='both', linestyle='--')
    # 设置y轴范围，让低准确率更明显
    ax1.set_ylim(0, 100)

    # --- 绘制损失曲线 (ax2) ---
    ax2.plot(df['epoch'], df['train_loss'], 'o-', label='Training Loss', color='C2')
    ax2.plot(df['epoch'], df['test_loss'], 's--', label='Validation Loss', color='C3')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('Training vs. Validation Loss')
    ax2.legend()
    ax2.grid(True, which='both', linestyle='--')

    # 优化布局并显示图表
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # 保存图表
    output_dir = os.path.dirname(log_file_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    output_image_path = os.path.join(output_dir, 'training_summary.png')
    plt.savefig(output_image_path, dpi=300, bbox_inches='tight')
    print(f"图表已保存至: {output_image_path}")

    plt.show()


if __name__ == '__main__':
    # 设置命令行参数解析
    parser = argparse.ArgumentParser(description='绘制训练日志图表')
    parser.add_argument('--log_file', type=str, 
                        default='./output/vim_tiny_A100_finetune_100_epochs/log.txt',
                        help='训练日志文件路径')
    
    args = parser.parse_args()
    
    # 调用绘图函数
    plot_training_log(args.log_file)