#!/bin/bash
cd /root/ViM/Vim-main/vim;

CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --standalone main.py \
    --model hier_vim_tiny  \
    --batch-size 256 \
    --epochs 200 \
    --drop-path 0.1 \
    --weight-decay 0.1 \
    --num_workers 25 \
    --lr 5e-4 \
    --warmup-epochs 20 \
    --cooldown-epochs 5 \
    --data-path /root/autodl-tmp/ImageNet \
    --output_dir ./output/hiervim_tiny_200_epoch \
    --if_amp \
    --mixup 0.8 \
    --cutmix 1.0 \
    --smoothing 0.1 \
    --clip-grad 1.0 \
    --resume ./output/hiervim_tiny_200_epoch/checkpoint.pth
    