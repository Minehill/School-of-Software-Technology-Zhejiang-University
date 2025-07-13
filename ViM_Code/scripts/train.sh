cd /root/ViM/Vim-main/vim;

CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --standalone main.py \
    --model vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2 \
    --batch-size 128 \
    --epochs 100 \
    --drop-path 0.0 \
    --weight-decay 0.1 \
    --num_workers 25 \
    --lr 1e-3 \
    --data-path /root/autodl-tmp/ImageNet \
    --output_dir ./output/vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2 \
    --no_amp