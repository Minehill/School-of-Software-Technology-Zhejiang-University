# 权重文件百度网盘链接
通过网盘分享的文件：浙软数字孪生任务2-子任务1
链接: https://pan.baidu.com/s/1gQzJWCs4ztCe2wqQep49ZA?pwd=1234 提取码: 1234

# 对应关系
├─cifar100_sdt_v3_scratch: cifar100数据集上复现情况  
├─cifar10_sdt_v3_scratch：cifar10数据集上复现情况  
├─ERR_IMNET_ATAN:误差控制的ATAN代替梯度  
├─ERR_IMNET_ATAN：误差控制的矩形代替梯度  
├─IMNET_ATAN：ATAN代替梯度  
├─IMNET_sdt_v3_scratch：ImageNet子集上复现情况  
├─Learn_Layer_Norm：可学习归一化参数  
├─Learn_MaxValue：可学习边界D值（未实现）  
├─PositionEncode：位置编码  
└─trapezoid：梯形代替梯度  

# 代码说明
1. 复现的终端指令为：
    ```
    CIFAT10:
    torchrun --standalone --nproc_per_node=1 \
    main_finetune.py \
    --model Efficient_Spiking_Transformer_s \
    --data_set CIFAR10 \
    --data_path ./dataset/CIFAR-10/ \
    --nb_classes 10 \
    --batch_size 128 \
    --blr 5e-4 \
    --epochs 50 \
    --warmup_epochs 20 \
    --output_dir ./outputs/cifar10_sdt_v3_scratch \
    --log_dir ./logs/cifar10_sdt_v3_scratch \
    --dist_eval 
    ```
    ```
    CIFAR100:
    torchrun --standalone --nproc_per_node=1 \
    main_finetune.py \
    --model Efficient_Spiking_Transformer_s \
    --data_set CIFAR100 \
    --data_path ./dataset/CIFAR-100/ \
    --nb_classes 10 \
    --batch_size 256 \
    --blr 5e-3 \
    --epochs 100 \
    --warmup_epochs 20 \
    --output_dir ./outputs/cifar100_sdt_v3_scratch \
    --log_dir ./logs/cifar100_sdt_v3_scratch \
    --dist_eval 
    ```
    ```
    IMNET
    torchrun --standalone --nproc_per_node=1 \
    main_finetune.py \
    --model Efficient_Spiking_Transformer_s \
    --data_set IMNET \
    --data_path ./dataset/ImageNet/ \
    --nb_classes 10 \
    --batch_size 256 \
    --blr 5e-3 \
    --epochs 100 \
    --warmup_epochs 20 \
    --output_dir ./outputs/IMNET_sdt_v3_scratch \
    --log_dir ./logs/IMNET_sdt_v3_scratch \
    --dist_eval 
    ```
2. 数据集下载：
[ImageNet-1K子集](https://blog.csdn.net/m0_46412065/article/details/128724252)、
[CIFAR-10/100](https://www.cs.toronto.edu/~kriz/cifar.html)
3. 每一个优化方向都有一个不同的models.py,由于时间紧张，我没有将这些整合进同一个文件中，如果需要运行相关代码，请将对应的modelsxx.py文件重命名为models.py文件。
4. 对于替代梯度相关的优化,将models_替代梯度选择.py重名名，并将./util/datasets_替代梯度选择.py重重名为datasets.py。运行指令：
    ```
    torchrun --standalone --nproc_per_node=1 \
    main_finetune.py \
    --model Efficient_Spiking_Transformer_s \
    --data_set IMNET \
    --data_path ./dataset/ImageNet/ \
    --nb_classes 10 \
    --batch_size 256 \
    --blr 5e-3 \
    --epochs 100 \
    --warmup_epochs 20 \
    --output_dir ./outputs/指定存储目录 \
    --log_dir ./logs/指定存储目录 \
    --dist_eval \
    --surrogate_gradient_type 选择代替梯度(ori为原代替梯度) \
    ```
5. 其他的优化选择请在对应文件夹内找到modelsxx.py
