import torch
import argparse
from fvcore.nn import FlopCountAnalysis, parameter_count_table

import models 

def analyze_model(model_name, input_size=224, timesteps=4):
    """
    分析模型的参数量、FLOPs和吞-吐量。
    """
    print(f"--- Analyzing Model: {model_name} ---")
    print(f"Input size: {input_size}x{input_size}, Timesteps: {timesteps}")


    try:
        model = models.__dict__[model_name]()
        if hasattr(model, 'T'):
            model.T = timesteps
        model.eval()
        print("Model created successfully.")
    except Exception as e:
        print(f"Error creating model: {e}")
        return

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[1] Parameters")
    print(f"Total trainable parameters: {param_count / 1e6:.2f} M")

    print(f"\n[2] FLOPs (Floating Point Operations)")
    try:
        # 将 T 设为 1，分析单步的计算量
        if hasattr(model, 'T'):
            model.T = 1
        
        # 创建一个 dummy input
        dummy_input = torch.randn(1, 3, input_size, input_size)
        
        # 使用 fvcore 进行分析
        flops = FlopCountAnalysis(model, dummy_input)
        total_flops = flops.total()
        
        print(f"FLOPs for a single timestep (T=1): {total_flops / 1e9:.2f} GFLOPs")
        print(f"Estimated total FLOPs for T={timesteps}: {(total_flops * timesteps) / 1e9:.2f} GFLOPs")


    except Exception as e:
        print(f"Could not calculate FLOPs. This might be due to unsupported operations in the model for fvcore. Error: {e}")


    print(f"\n[3] Throughput (images/sec)")
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        dummy_input = torch.randn(32, 3, input_size, input_size, device=device) 

        # 预热
        print("Warming up...")
        for _ in range(20):
            with torch.no_grad():
                _ = model(dummy_input)
        
        # 正式测量
        print("Measuring...")
        torch.cuda.synchronize()
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        
        start_time.record()
        # 运行多次取平均
        num_runs = 100
        for _ in range(num_runs):
            with torch.no_grad():
                _ = model(dummy_input)
        end_time.record()
        torch.cuda.synchronize()
        
        elapsed_time_ms = start_time.elapsed_time(end_time)
        throughput = (num_runs * dummy_input.size(0)) / (elapsed_time_ms / 1000.0)
        
        print(f"Hardware: {torch.cuda.get_device_name(device)}")
        print(f"Time for {num_runs} runs: {elapsed_time_ms:.2f} ms")
        print(f"Throughput: {throughput:.2f} images/sec")
        
    except Exception as e:
        print(f"Could not measure throughput. Error: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Model Efficiency Analysis')
    parser.add_argument('--model', default='SpikeMambaFormer_CIFAR10_model', type=str, help='Name of model to analyze')
    parser.add_argument('--input-size', default=32, type=int, help='Input image size')
    parser.add_argument('--timesteps', default=4, type=int, help='SNN simulation timesteps')
    args = parser.parse_args()

    analyze_model(args.model, args.input_size, args.timesteps)