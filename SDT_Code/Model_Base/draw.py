import json
import matplotlib.pyplot as plt
import argparse
import os

def plot_metrics(log_path, save_dir):
    """
    从 log.txt 文件中读取训练指标并绘制图表。

    Args:
        log_path (str): log.txt 文件的路径。
        save_dir (str): 图表保存的目录。
    """
    epochs = []
    train_losses = []
    test_losses = []
    test_acc1s = []
    test_acc5s = []

    try:
        with open(log_path, 'r') as f:
            for line in f:
                log_data = json.loads(line)
                if 'epoch' in log_data and 'train_loss' in log_data and 'test_loss' in log_data and 'test_acc1' in log_data:
                    epochs.append(log_data['epoch'])
                    train_losses.append(log_data['train_loss'])
                    test_losses.append(log_data['test_loss'])
                    test_acc1s.append(log_data['test_acc1'])
                    test_acc5s.append(log_data['test_acc5'])
    except FileNotFoundError:
        print(f"Error: Log file not found at {log_path}")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {log_path}. The file might be corrupted or empty.")
        return
        
    if not epochs:
        print("No data found in log file to plot.")
        return

    # --- 开始绘图 ---
    plt.style.use('seaborn-v0_8-whitegrid') 
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # 图1: 损失曲线
    ax1.plot(epochs, train_losses, label='Train Loss', color='dodgerblue', marker='o', linestyle='-')
    ax1.plot(epochs, test_losses, label='Validation Loss', color='darkorange', marker='s', linestyle='--')
    ax1.set_title('Training and Validation Loss', fontsize=16)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.tick_params(axis='both', which='major', labelsize=10)

    # 图2: 精度曲线
    ax2.plot(epochs, test_acc1s, label='Validation Acc@1', color='forestgreen', marker='o', linestyle='-')
    ax2.plot(epochs, test_acc5s, label='Validation Acc@5', color='crimson', marker='s', linestyle='--')
    ax2.set_title('Validation Accuracy', fontsize=16)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.tick_params(axis='both', which='major', labelsize=10)

    # 找到最佳 Acc@1 并标记
    best_acc1 = max(test_acc1s)
    best_epoch = epochs[test_acc1s.index(best_acc1)]
    ax2.axhline(y=best_acc1, color='gray', linestyle=':', linewidth=1.5)
    ax2.axvline(x=best_epoch, color='gray', linestyle=':', linewidth=1.5)
    ax2.annotate(f'Best Acc@1: {best_acc1:.2f}% at Epoch {best_epoch}',
                 xy=(best_epoch, best_acc1),
                 xytext=(best_epoch, best_acc1 - 10),
                 arrowprops=dict(facecolor='black', shrink=0.05),
                 fontsize=10,
                 bbox=dict(boxstyle="round,pad=0.3", fc="yellow", ec="black", lw=1, alpha=0.7))


    plt.tight_layout()

    save_path = os.path.join(save_dir, 'training_metrics.png')
    plt.savefig(save_path, dpi=300)
    print(f"Plot saved to {save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot training metrics from log file.')
    parser.add_argument('--log_file', type=str, required=True,
                        help='Path to the log.txt file.')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='Directory to save the plot. Defaults to the log file\'s directory.')
    
    args = parser.parse_args()

    if args.save_dir is None:
        args.save_dir = os.path.dirname(args.log_file)
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    plot_metrics(args.log_file, args.save_dir)