# 权重文件网盘链接
通过网盘分享的文件：浙软数字孪生任务2-子任务3
链接: https://pan.baidu.com/s/1V82xq99bpFXJOkgNI-qukA?pwd=1234 提取码: 1234

# 训练
```
torchrun --standalone --nproc_per_node=1 \
  main_finetune.py \
  --model Efficient_Spiking_Transformer_s \
  --data_path ./dataset/ \
  --nb_classes 10 \
  --batch_size 256 \
  --blr 5e-3 \
  --epochs 100 \
  --warmup_epochs 20 \
  --output_dir ./outputs/CIFAR10_SNN-Mamba \
  --log_dir ./logs/CIFAR10-Mamba \
  --dist_eval 
```

# 效率评估
将analyze_model.py中的import models 改为 import 模型定义.py as models，然后运行
```
python analyze_model.py --model Efficient_Spiking_Transformer_s --input-size 32 --timesteps 4
```