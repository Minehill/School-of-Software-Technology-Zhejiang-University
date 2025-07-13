import torch
import torchinfo
import torch.nn as nn
from timm.models.layers import to_2tuple, trunc_normal_, DropPath
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
from einops.layers.torch import Rearrange
import torch.nn.functional as F
from functools import partial

import os
import math

class Quant(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, i, min_value, max_value, surrogate_gradient_type='atan', alpha=2.0, k=1.0):
        # 保存所有反向传播需要的参数
        ctx.min = min_value
        ctx.max = max_value
        ctx.surrogate_gradient_type = surrogate_gradient_type
        ctx.alpha = alpha
        ctx.k = k
        # 保存前向传播的输入和输出，用于反向传播计算
        rounded_i = torch.round(torch.clamp(i, min=min_value, max=max_value))
        ctx.save_for_backward(i, rounded_i)
        
        return rounded_i

    @staticmethod
    @torch.cuda.amp.custom_fwd
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        i, rounded_i = ctx.saved_tensors
        
        # 只在[min, max]区间内计算梯度
        mask = (i >= ctx.min) & (i <= ctx.max)
        
        if ctx.surrogate_gradient_type == 'ori': 
            grad_factor = torch.ones_like(i)
            
        elif ctx.surrogate_gradient_type == 'atan':
            center = (ctx.max + ctx.min) / 2.0
            width = (ctx.max - ctx.min) / 2.0
            x = (i - center) / width if width > 0 else torch.zeros_like(i)
            grad_factor = ctx.alpha / 2.0 / (1 + (math.pi / 2.0 * ctx.alpha * x).pow(2))

        elif ctx.surrogate_gradient_type == 'error_comp_atan':
            # 基于ATan的误差补偿梯度
            center = (ctx.max + ctx.min) / 2.0
            width = (ctx.max - ctx.min) / 2.0
            x = (i - center) / width if width > 0 else torch.zeros_like(i)
            base_grad = ctx.alpha / 2.0 / (1 + (math.pi / 2.0 * ctx.alpha * x).pow(2))
            
            quantization_error = torch.abs(i - rounded_i)
            comp_factor = 1 + ctx.k * quantization_error
            grad_factor = base_grad * comp_factor

        elif ctx.surrogate_gradient_type == 'error_weighted_rect':
            # 基础梯度为1
            center = (ctx.max + ctx.min) / 2.0
            width = (ctx.max - ctx.min) / 2.0
            x = (i - center) / width if width > 0 else torch.zeros_like(i)
            base_grad = torch.ones_like(i)
            quantization_error_norm = torch.abs(i - rounded_i)
            comp_factor = 1 + quantization_error_norm
            grad_factor = base_grad * comp_factor

        elif ctx.surrogate_gradient_type == 'trapezoid':
            center = (ctx.max + ctx.min) / 2.0
            width = (ctx.max - ctx.min)
            # 将膜电位i归一化到[-0.5, 0.5]区间
            x = (i - center) / width if width > 0 else torch.zeros_like(i)       
            # alpha 在这里被复用为中心平台的宽度比例 (0, 1]
            platform_width_ratio = ctx.alpha 
            
            # 计算归一化坐标下的平台半宽度
            half_platform = platform_width_ratio / 2.0
            
            # 创建一个全1的梯度因子
            grad_factor = torch.ones_like(x)
            
            # 找到斜坡区域
            slope_region_mask = torch.abs(x) > half_platform
            
            # 在斜坡区域计算线性下降的梯度
            # 斜坡的起始点是half_platform，结束点是0.5
            slope_width = 0.5 - half_platform
            if slope_width > 1e-6: # 避免除以0
                # |x|从half_platform变化到0.5，梯度从1变化到0
                slope_grad = (0.5 - torch.abs(x)[slope_region_mask]) / slope_width
                grad_factor[slope_region_mask] = slope_grad
        
        else:
            raise NotImplementedError(f"Surrogate gradient type '{ctx.surrogate_gradient_type}' not implemented.")

        final_grad = grad_input * grad_factor * mask
        
        return final_grad, None, None, None, None, None

class MultiSpike(nn.Module):
    def __init__(
        self,
        min_value=0,
        max_value=4,
        Norm=None,
        surrogate_gradient_type='atan',
        alpha=2.0, # alpha for atan
        k=1.0,     # k for error compensation
        ):
        super().__init__()
        if Norm is None:
            self.Norm = max_value
        else:
            self.Norm = Norm
        self.min_value = min_value
        self.max_value = max_value
        self.surrogate_gradient_type = surrogate_gradient_type
        self.alpha = alpha
        self.k = k
    
    def forward(self, x):
        quantized_x = Quant.apply(
            x, 
            self.min_value, 
            self.max_value, 
            self.surrogate_gradient_type, 
            self.alpha, 
            self.k
        )
        return quantized_x / self.Norm
        
    def __repr__(self):
        return (f"MultiSpike(Max_Value={self.max_value}, Min_Value={self.min_value}, Norm={self.Norm}, "
                f"SG_Type='{self.surrogate_gradient_type}', alpha={self.alpha}, k={self.k})")

class SepConv_Spike(nn.Module):
    def __init__(
        self,
        dim,
        expansion_ratio=2,
        bias=False,
        kernel_size=7,
        padding=3,
        **kwargs,
    ):
        super().__init__()
        med_channels = int(expansion_ratio * dim)
        self.spike1 = MultiSpike(**kwargs)
        self.pwconv1 = nn.Sequential(
            nn.Conv2d(dim, med_channels, kernel_size=1, stride=1, bias=bias),
            nn.BatchNorm2d(med_channels)
            )
        self.spike2 = MultiSpike(**kwargs)
        self.dwconv = nn.Sequential(
            nn.Conv2d(med_channels, med_channels, kernel_size=kernel_size, padding=padding, groups=med_channels, bias=bias),
            nn.BatchNorm2d(med_channels)
        )
        self.spike3 = MultiSpike(**kwargs)
        self.pwconv2 = nn.Sequential(
            nn.Conv2d(med_channels, dim, kernel_size=1, stride=1, bias=bias),
            nn.BatchNorm2d(dim)
        )

    def forward(self, x):
        x = self.spike1(x)
        x = self.pwconv1(x)
        x = self.spike2(x)
        x = self.dwconv(x)
        x = self.spike3(x)
        x = self.pwconv2(x)
        return x

class MS_ConvBlock_spike_SepConv(nn.Module):
    def __init__(
        self,
        dim,
        mlp_ratio=4.0,
        **kwargs,
    ):
        super().__init__()

        self.Conv = SepConv_Spike(dim=dim, **kwargs)
        self.mlp_ratio = mlp_ratio
        self.spike1 = MultiSpike(**kwargs)
        self.conv1 = nn.Conv2d(dim, int(dim * mlp_ratio), kernel_size=3, padding=1, groups=1, bias=False)
        self.bn1 = nn.BatchNorm2d(int(dim * mlp_ratio)) 
        self.spike2 = MultiSpike(**kwargs)
        self.conv2 = nn.Conv2d(int(dim * mlp_ratio), dim, kernel_size=3, padding=1, groups=1, bias=False)
        self.bn2 = nn.BatchNorm2d(dim)  

    def forward(self, x):
        B, C, H, W = x.shape
        x_feat = self.Conv(x) + x
        x_mlp = self.spike1(x_feat)
        x_mlp = self.bn1(self.conv1(x_mlp))
        x_mlp = self.spike2(x_mlp)
        x_mlp = self.bn2(self.conv2(x_mlp))
        x = x_feat + x_mlp
        return x

class MS_Attention_linear(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        lamda_ratio=1,
        **kwargs,
    ):
        super().__init__()
        assert (dim % num_heads == 0), f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        self.scale = (dim//num_heads) ** -0.5
        self.lamda_ratio = lamda_ratio
        
        self.head_spike = MultiSpike(**kwargs)
        self.q_conv = nn.Sequential(nn.Conv2d(dim, dim, 1, 1, bias=qkv_bias), nn.BatchNorm2d(dim))
        self.k_conv = nn.Sequential(nn.Conv2d(dim, dim, 1, 1, bias=qkv_bias), nn.BatchNorm2d(dim))
        self.v_conv = nn.Sequential(nn.Conv2d(dim, int(dim*lamda_ratio), 1, 1, bias=qkv_bias), nn.BatchNorm2d(int(dim*lamda_ratio)))
        
        self.q_spike = MultiSpike(**kwargs)
        self.k_spike = MultiSpike(**kwargs)
        self.v_spike = MultiSpike(**kwargs)
        self.attn_spike = MultiSpike(**kwargs)

        self.proj_conv = nn.Sequential(
            nn.Conv2d(int(dim*lamda_ratio), dim, 1, 1, bias=qkv_bias), nn.BatchNorm2d(dim)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
        C_v = int(C*self.lamda_ratio)

        x_spiked = self.head_spike(x)
        q = self.q_conv(x_spiked)
        k = self.k_conv(x_spiked)
        v = self.v_conv(x_spiked)

        q = self.q_spike(q).flatten(2).transpose(-1, -2).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k_spike(k).flatten(2).transpose(-1, -2).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v_spike(v).flatten(2).transpose(-1, -2).reshape(B, N, self.num_heads, C_v // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        x = (attn @ v).transpose(1, 2).reshape(B, N, C_v)
        x = x.transpose(-1, -2).reshape(B, C_v, H, W)
        
        x = self.attn_spike(x)
        x = self.proj_conv(x)
        return x

class MS_Block_Spike_SepConv(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        drop_path=0.0,
        **kwargs,
    ):
        super().__init__()
        self.conv = SepConv_Spike(dim=dim, kernel_size=3, padding=1, **kwargs)
        self.attn = MS_Attention_linear(dim, num_heads=num_heads, lamda_ratio=4, **kwargs)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = self.build_mlp(dim, mlp_hidden_dim, **kwargs)

    def build_mlp(self, in_features, hidden_features, **kwargs):
        return nn.Sequential(
            MultiSpike(**kwargs),
            nn.Conv1d(in_features, hidden_features, kernel_size=1),
            nn.BatchNorm1d(hidden_features),
            MultiSpike(**kwargs),
            nn.Conv1d(hidden_features, in_features, kernel_size=1),
            nn.BatchNorm1d(in_features),
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x = x + self.drop_path(self.conv(x))
        x = x + self.drop_path(self.attn(x))
        # MLP part needs reshape
        x_mlp = x.flatten(2)
        x_mlp = self.mlp(x_mlp)
        x = x + self.drop_path(x_mlp.reshape(B, C, H, W))
        return x

class MS_DownSampling(nn.Module):
    def __init__(
        self,
        in_channels,
        embed_dims,
        kernel_size=3,
        stride=2,
        padding=1,
        first_layer=False,
        **kwargs,
    ):
        super().__init__()
        self.encode_conv = nn.Conv2d(in_channels, embed_dims, kernel_size=kernel_size, stride=stride, padding=padding)
        self.encode_bn = nn.BatchNorm2d(embed_dims)
        self.first_layer = first_layer
        if not self.first_layer:
            self.encode_spike = MultiSpike(**kwargs)

    def forward(self, x):
        if not self.first_layer:
            x = self.encode_spike(x)
        x = self.encode_conv(x)
        x = self.encode_bn(x)
        return x

class Spiking_vit_MetaFormer_Spike_SepConv(nn.Module):
    def __init__(
        self,
        in_channels=3,
        num_classes=1000,
        embed_dim=[64, 128, 256],
        num_heads=[1, 2, 4],
        mlp_ratios=[4, 4, 4],
        depths=[6, 8, 6],
        drop_path_rate=0.0,
        **kwargs, 
    ):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        
        spike_kwargs = kwargs

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # Stage 1
        self.downsample1_1 = MS_DownSampling(in_channels=in_channels, embed_dims=embed_dim[0] // 2, kernel_size=7, stride=2, padding=3, first_layer=True, **spike_kwargs)
        self.ConvBlock1_1 = nn.ModuleList([MS_ConvBlock_spike_SepConv(dim=embed_dim[0] // 2, mlp_ratio=mlp_ratios[0], **spike_kwargs)])
        self.downsample1_2 = MS_DownSampling(in_channels=embed_dim[0] // 2, embed_dims=embed_dim[0], kernel_size=3, stride=2, padding=1, **spike_kwargs)
        self.ConvBlock1_2 = nn.ModuleList([MS_ConvBlock_spike_SepConv(dim=embed_dim[0], mlp_ratio=mlp_ratios[0], **spike_kwargs)])

        # Stage 2
        self.downsample2 = MS_DownSampling(in_channels=embed_dim[0], embed_dims=embed_dim[1], **spike_kwargs)
        self.ConvBlock2_1 = nn.ModuleList([MS_ConvBlock_spike_SepConv(dim=embed_dim[1], mlp_ratio=mlp_ratios[1], **spike_kwargs) for _ in range(2)]) # Replicated from original code

        # Stage 3
        self.downsample3 = MS_DownSampling(in_channels=embed_dim[1], embed_dims=embed_dim[2], **spike_kwargs)
        self.block3 = nn.ModuleList([MS_Block_Spike_SepConv(dim=embed_dim[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], drop_path=dpr[i], **spike_kwargs) for i in range(depths[0])])

        # Stage 4
        self.downsample4 = MS_DownSampling(in_channels=embed_dim[2], embed_dims=embed_dim[3], stride=1, **spike_kwargs) # stride 1
        self.block4 = nn.ModuleList([MS_Block_Spike_SepConv(dim=embed_dim[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], drop_path=dpr[i + depths[0]], **spike_kwargs) for i in range(depths[1])])
        
        self.head = nn.Linear(embed_dim[3], num_classes) if num_classes > 0 else nn.Identity()
        self.spike = MultiSpike(Norm=1, **spike_kwargs)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.Conv1d, nn.Conv2d)):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.downsample1_1(x)
        for blk in self.ConvBlock1_1: x = blk(x)
        x = self.downsample1_2(x)
        for blk in self.ConvBlock1_2: x = blk(x)
        
        x = self.downsample2(x)
        for blk in self.ConvBlock2_1: x = blk(x)

        x = self.downsample3(x)
        for blk in self.block3: x = blk(x)
        
        x = self.downsample4(x)
        for blk in self.block4: x = blk(x)
        
        x = x.flatten(2).mean(2)
        x = self.spike(x)
        x = self.head(x)
        return x

def Efficient_Spiking_Transformer_s(**kwargs):
    model = Spiking_vit_MetaFormer_Spike_SepConv(
        embed_dim=[32, 64, 128, 192],
        num_heads=[8, 8, 8, 8],
        mlp_ratios=[4, 4, 4, 4],
        depths=[6, 2, 0, 0], # Simplified based on original structure for 's' model
        **kwargs,
    )
    return model

def Efficient_Spiking_Transformer_m(**kwargs):
    model = Spiking_vit_MetaFormer_Spike_SepConv(
        embed_dim=[48, 96, 192, 240],
        num_heads=[8, 8, 8, 8],
        mlp_ratios=[4, 4, 4, 4],
        depths=[6, 2, 0, 0],
        **kwargs,
    )
    return model

def Efficient_Spiking_Transformer_l(**kwargs):
    model = Spiking_vit_MetaFormer_Spike_SepConv(
        embed_dim=[64, 128, 256, 360],
        num_heads=[8, 8, 8, 8],
        mlp_ratios=[4, 4, 4, 4],
        depths=[8, 2, 0, 0],
        **kwargs,
    )
    return model

def Efficient_Spiking_Transformer_t(**kwargs):
    model = Spiking_vit_MetaFormer_Spike_SepConv(
        embed_dim=[24, 48, 96, 128],
        num_heads=[8, 8, 8, 8],
        mlp_ratios=[4, 4, 4, 4],
        depths=[8, 2, 0, 0],
        **kwargs,
    )
    return model

if __name__ == "__main__":
    model = Efficient_Spiking_Transformer_s(surrogate_gradient_type='atan', alpha=2.5)
    print(model)
    x = torch.randn(1, 3, 224, 224)
    y = model(x)
    print("Output shape:", y.shape)
    torchinfo.summary(model, (1, 3, 224, 224), device='cpu')