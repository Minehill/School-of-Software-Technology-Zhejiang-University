# analyze_grad_log.py (健壮版)

import pandas as pd
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import re # 导入正则表达式库

def analyze_gradients(log_file_path_str: str):
    """
    读取并分析梯度日志文件，使用正则表达式使其更健壮。
    """
    log_file = Path(log_file_path_str)
    print(f"Analyzing log file: {log_file}\n")
    
    # E:(整数), S:(整数), P:(任意字符), GradNorm:(浮点数或NaN/Inf)
    line_regex = re.compile(r"E:(\d+), S:(\d+), P:(.+?), GradNorm:(.+)")

    parsed_data = []
    try:
        with open(log_file, 'r') as f:
            for line in f:
                # 忽略空行
                if not line.strip():
                    continue
                
                match = line_regex.match(line.strip())
                if match:
                    epoch, step, param, grad_norm_str = match.groups()
                    
                    # 处理 grad_norm 可能为 NaN 或 Inf 的情况
                    if "NaN" in grad_norm_str or "Inf" in grad_norm_str:
                        grad_norm = float('nan') # 用 NaN 表示
                    else:
                        grad_norm = float(grad_norm_str)
                        
                    parsed_data.append([int(epoch), int(step), param, grad_norm])

        if not parsed_data:
            print("No valid data found in the log file.")
            return
            
        # 将解析好的数据转换为 DataFrame
        df = pd.DataFrame(parsed_data, columns=['Epoch', 'Step', 'Param', 'GradNorm'])
        # 去掉梯度为 NaN 的行，以便进行数学计算
        df = df.dropna(subset=['GradNorm'])

    except Exception as e:
        print(f"Error reading or parsing the file: {e}")
        return

    if df.empty:
        print("DataFrame is empty after processing. No valid gradient data to analyze.")
        return
    
    # 1. 找到梯度最大的 Top 10 条记录
    print("--- Top 10 Max Gradient Norm Records ---")
    top_10_max_grads = df.sort_values(by='GradNorm', ascending=False).head(10)
    print(top_10_max_grads)
    print("-" * 40)

    # 2. 按参数名聚合
    param_summary = df.groupby('Param')['GradNorm'].agg(['mean', 'max', 'count']).sort_values(by='max', ascending=False)
    print("\n--- Gradient Norm Summary per Parameter (sorted by max norm) ---")
    print(param_summary.head(20))
    print("-" * 40)

    # 3. 按 Epoch 和 Step 聚合
    time_summary = df.groupby(['Epoch', 'Step'])['GradNorm'].sum()
    print(f"\n--- Total Gradient Norm per Step (First 10 steps) ---")
    print(time_summary.head(10))
    print("-" * 40)

    # 4. 可视化
    print("\nGenerating plots...")
    output_dir = log_file.parent
    
    # a. 参数梯度条形图
    plt.figure(figsize=(12, 8))
    top_params = param_summary.head(20)
    sns.barplot(x=top_params['max'], y=top_params.index)
    plt.title('Top 20 Max Gradient Norm per Parameter')
    plt.xlabel('Max Gradient Norm (log scale)')
    plt.xscale('log')
    plt.tight_layout()
    plt.savefig(output_dir / 'max_grad_norm_per_param.png')
    print(f"Saved plot: {output_dir / 'max_grad_norm_per_param.png'}")
    
    # b. 梯度和随时间变化趋势图
    plt.figure(figsize=(12, 6))
    time_summary.plot()
    plt.title('Total Gradient Norm Sum over Time (Epoch, Step)')
    plt.xlabel('Step')
    plt.ylabel('Sum of Gradient Norms (log scale)')
    plt.yscale('log')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / 'total_grad_norm_over_time.png')
    print(f"Saved plot: {output_dir / 'total_grad_norm_over_time.png'}")
    
    plt.close('all')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze Gradient Log File.')
    parser.add_argument('log_file', type=str, help='Path to the gradient log file.')
    args = parser.parse_args()
    
    analyze_gradients(args.log_file)