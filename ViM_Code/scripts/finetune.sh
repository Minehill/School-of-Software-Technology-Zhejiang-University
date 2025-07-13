#!/bin/bash
cd /root/ViM/Vim-main/vim;

CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --standalone main.py \
    --model vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2 \
    --batch-size 256 \
    --epochs 100 \
    --drop-path 0.1 \
    --weight-decay 0.05 \
    --num_workers 25 \
    --lr 2e-3 \
    --warmup-epochs 5 \
    --cooldown-epochs 5 \
    --data-path /root/autodl-tmp/ImageNet \
    --output_dir ./output/vim_tiny_A100_finetune_100_epochs \
    --if_amp \
    --mixup 0.8 \
    --cutmix 1.0 \
    --smoothing 0.1 \
    --finetune ./output/vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2/checkpoint.pth