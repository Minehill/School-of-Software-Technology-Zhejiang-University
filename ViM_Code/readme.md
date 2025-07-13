# 权重模型百度网盘链接
通过网盘分享的文件：浙软数字孪生任务2-子任务2
链接: https://pan.baidu.com/s/1dFHrTW3I-twuLi56czVySg?pwd=1234 提取码: 1234

# 代码说明
1. 复现直接执行```bash ./scripts/train.sh```进行第一个100轮的训练，执行```bash ./scripts/finetune.sh```在train的基础上进行第二个100轮的训练
2. 如果要运行四向扫描优化代码，首先将models_mamba_四向扫描优化.py重命名为models_mamba.py，然后执行```bash ./scripts/fourwayscan.sh```
3. 如果要运行多尺度优化代码，首先将main_多尺度优化.py重命名为main.py,然后执行```bash ./scripts/hierarchical_mamba.sh```即可。若要取消梯度追踪请将main中第385行的```TRACK_GRADIENTS = True```设置为False

4. output的目录树：  
├─hiervim_tiny_200_epoch：多尺度优化  
├─vim_tiny_4way_scan_from_scratch：四向扫描优化  
├─vim_tiny_A100_finetune_100_epochs：第二个100轮/最终复现结果  
└─vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2：第一个100轮初次训练结果  

5. draw.py根据日志绘制图表；analy_grad.py根据梯度日志绘制图表
6. 时间紧张没来得及将整个项目彻底整理，敬请见谅