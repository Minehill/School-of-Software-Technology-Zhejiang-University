import torch

# 加载checkpoint
checkpoint = torch.load('checkpoint-0.pth')

print("Checkpoint keys:", checkpoint.keys())

if "model_state_dict" in checkpoint:
    state_dict = checkpoint["model_state_dict"]
elif "state_dict" in checkpoint:
    state_dict = checkpoint["state_dict"]
elif "model" in checkpoint:
    state_dict = checkpoint["model"]
else:
    print("无法自动识别模型状态字典，请检查打印的键列表")
    state_dict = checkpoint  

# 计算参数总量
total_params = sum(p.numel() for p in state_dict.values() if isinstance(p, torch.Tensor))
print(f"模型总参数量: {total_params}")
print(f"模型参数量(M): {total_params / 1_000_000:.2f}M") 
print(f"模型参数量(MB): {total_params * 4 / (1024 * 1024):.2f} MB")  # 假设float32类型(4字节)