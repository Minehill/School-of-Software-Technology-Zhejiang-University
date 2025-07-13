# models.py (替换版: SpikeMambaFormer)

import torch
import torchinfo
import torch.nn as nn
from timm.models.layers import to_2tuple, trunc_normal_, DropPath
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
from einops.layers.torch import Rearrange
import torch.nn.functional as F
from functools import partial

# =========================================================================
from mamba_ssm.modules.mamba_simple import Mamba
# =========================================================================

import os


class Quant(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, i, min_value, max_value):
        ctx.min = min_value
        ctx.max = max_value
        ctx.save_for_backward(i)
        return torch.round(torch.clamp(i, min=min_value, max=max_value))

    @staticmethod
    @torch.cuda.amp.custom_fwd
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        i, = ctx.saved_tensors
        grad_input[i < ctx.min] = 0
        grad_input[i > ctx.max] = 0
        return grad_input, None, None

class MultiSpike(nn.Module):
    def __init__(self, min_value=0, max_value=4, Norm=None):
        super().__init__()
        if Norm is None: self.Norm = max_value
        else: self.Norm = Norm
        self.min_value = min_value
        self.max_value = max_value
    
    @staticmethod
    def spike_function(x, min_value, max_value):
        return Quant.apply(x, min_value, max_value)
        
    def __repr__(self):
        return f"MultiSpike(Max_Value={self.max_value}, Min_Value={self.min_value}, Norm={self.Norm})"     

    def forward(self, x):
        return self.spike_function(x, min_value=self.min_value, max_value=self.max_value) / self.Norm



class SepConv_Spike(nn.Module):
    def __init__(self, dim, expansion_ratio=2, bias=False, kernel_size=7, padding=3):
        super().__init__()
        med_channels = int(expansion_ratio * dim)
        self.spike1 = MultiSpike()
        self.pwconv1 = nn.Sequential(nn.Conv2d(dim, med_channels, 1, 1, bias=bias), nn.BatchNorm2d(med_channels))
        self.spike2 = MultiSpike()
        self.dwconv = nn.Sequential(nn.Conv2d(med_channels, med_channels, kernel_size, padding=padding, groups=med_channels, bias=bias), nn.BatchNorm2d(med_channels))
        self.spike3 = MultiSpike()
        self.pwconv2 = nn.Sequential(nn.Conv2d(med_channels, dim, 1, 1, bias=bias), nn.BatchNorm2d(dim))

    def forward(self, x):
        x = self.spike1(x)
        x = self.pwconv1(x)
        x = self.spike2(x)
        x = self.dwconv(x)
        x = self.spike3(x)
        x = self.pwconv2(x)
        return x

class MS_MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1_conv = nn.Conv1d(in_features, hidden_features, 1, 1)
        self.fc1_bn = nn.BatchNorm1d(hidden_features)
        self.fc1_spike = MultiSpike()
        self.fc2_conv = nn.Conv1d(hidden_features, out_features, 1, 1)
        self.fc2_bn = nn.BatchNorm1d(out_features)
        self.fc2_spike = MultiSpike()
        self.c_hidden = hidden_features
        self.c_output = out_features

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
        x = x.flatten(2)
        x = self.fc1_spike(x)
        x = self.fc1_conv(x)
        x = self.fc1_bn(x)
        x = self.fc2_spike(x)
        x = self.fc2_conv(x)
        x = self.fc2_bn(x).reshape(B, self.c_output, H, W).contiguous()
        return x

# =========================================================================
# 创建 Mamba 分支，替换 Attention 分支
# =========================================================================

class Mamba_Branch(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.spike = MultiSpike()

    def forward(self, x):
        B, C, H, W = x.shape
        assert C == self.dim
        
        # 将输入从 [B, C, H, W] 转换为 [B, N, C]
        x_seq = x.flatten(2).transpose(1, 2)
        
        # LayerNorm
        x_seq = self.norm(x_seq)
        
        # Mamba 处理
        x_mamba = self.mamba(x_seq)
        
        # 将输出转换回 [B, C, H, W]
        x_out = x_mamba.transpose(1, 2).reshape(B, C, H, W)
        
        # 脉冲化
        x_out = self.spike(x_out)
        
        return x_out


class Mamba_Based_Block(nn.Module):
    """
    这是对原 MS_Block_Spike_SepConv 的替换。
    """
    def __init__(self, dim, mlp_ratio=4.0, drop_path=0.0, **kwargs): # 使用**kwargs接收多余参数
        super().__init__()
        
        self.conv = SepConv_Spike(dim=dim, kernel_size=3, padding=1)
        
        # 替换
        self.attn = Mamba_Branch(dim=dim)
        
        self.mlp = MS_MLP(in_features=dim, hidden_features=int(dim * mlp_ratio))
        
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        x = x + self.conv(x)
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x

class MS_DownSampling(nn.Module):
    def __init__(self, in_channels=2, embed_dims=256, kernel_size=3, stride=2, padding=1, first_layer=True):
        super().__init__()
        self.encode_conv = nn.Conv2d(in_channels, embed_dims, kernel_size, stride, padding)
        self.encode_bn = nn.BatchNorm2d(embed_dims)
        self.first_layer = first_layer
        if not first_layer: self.encode_spike = MultiSpike()

    def forward(self, x):
        if hasattr(self, "encode_spike"): x = self.encode_spike(x)
        x = self.encode_conv(x)
        x = self.encode_bn(x)
        return x

class MS_ConvBlock_spike_SepConv(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0):
        super().__init__()
        self.Conv = SepConv_Spike(dim=dim)
        self.mlp_ratio = mlp_ratio
        self.spike1 = MultiSpike()
        self.conv1 = nn.Conv2d(dim, dim * mlp_ratio, 3, 1, 1, groups=1, bias=False)
        self.bn1 = nn.BatchNorm2d(dim * mlp_ratio)
        self.spike2 = MultiSpike()
        self.conv2 = nn.Conv2d(dim * mlp_ratio, dim, 3, 1, 1, groups=1, bias=False)
        self.bn2 = nn.BatchNorm2d(dim)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.Conv(x) + x
        x_feat = x
        x = self.spike1(x)
        x = self.bn1(self.conv1(x)) # .reshape(B, self.mlp_ratio * C, H, W) bug? bn的输出就是4D
        x = self.spike2(x)
        x = self.bn2(self.conv2(x)) # .reshape(B, C, H, W)
        x = x_feat + x
        return x

# =========================================================================
#  主模型: 将 Transformer Block 替换为 Mamba Block
#  模型名保持 Spiking_vit_MetaFormer_Spike_SepConv 以兼容主程序
# =========================================================================
class SpikeMambaFormer(nn.Module):
    def __init__(self, img_size_h=128, img_size_w=128, in_channels=2, num_classes=10,
                 embed_dim=[64, 128, 256], num_heads=[1, 2, 4], mlp_ratios=[4, 4, 4],
                 drop_rate=0.0, drop_path_rate=0.0, norm_layer=nn.LayerNorm,
                 depths=[6, 8, 6], sr_ratios=[8, 4, 2], **kwargs): # **kwargs 接收所有参数
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        

        self.downsample1_1 = MS_DownSampling(in_channels=in_channels, embed_dims=embed_dim[0] // 2, kernel_size=7, stride=2, padding=3, first_layer=True)
        self.ConvBlock1_1 = nn.ModuleList([MS_ConvBlock_spike_SepConv(dim=embed_dim[0] // 2, mlp_ratio=mlp_ratios[0]) for _ in range(depths[0])])
        self.downsample1_2 = MS_DownSampling(in_channels=embed_dim[0] // 2, embed_dims=embed_dim[0], kernel_size=3, stride=2, padding=1, first_layer=False)
        
        self.ConvBlock2_1 = nn.ModuleList([MS_ConvBlock_spike_SepConv(dim=embed_dim[0], mlp_ratio=mlp_ratios[1]) for _ in range(depths[1])])
        self.downsample2 = MS_DownSampling(in_channels=embed_dim[0], embed_dims=embed_dim[1], kernel_size=3, stride=2, padding=1, first_layer=False)
        
        # Mamba-based 替换
        self.block3 = nn.ModuleList([Mamba_Based_Block(dim=embed_dim[1], mlp_ratio=mlp_ratios[2], drop_path=dpr[i]) for i in range(depths[2])])
        self.downsample3 = MS_DownSampling(in_channels=embed_dim[1], embed_dims=embed_dim[2], kernel_size=3, stride=2, padding=1, first_layer=False)
        
        # Mamba-based 替换
        self.block4 = nn.ModuleList([Mamba_Based_Block(dim=embed_dim[2], mlp_ratio=mlp_ratios[3], drop_path=dpr[i]) for i in range(depths[3])])
        
        self.head = nn.Linear(embed_dim[2], num_classes) if num_classes > 0 else nn.Identity()
        self.spike = MultiSpike(Norm=1)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        # 为了保持与原始模型一致的 block 数量，这里调整了循环
        x = self.downsample1_1(x)
        for blk in self.ConvBlock1_1: x = blk(x)
        x = self.downsample1_2(x)

        for blk in self.ConvBlock2_1: x = blk(x)
        x = self.downsample2(x)
        
        for blk in self.block3: x = blk(x)
        x = self.downsample3(x)

        for blk in self.block4: x = blk(x)
        
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = x.flatten(2).mean(2)
        x = self.spike(x)
        x = self.head(x)
        return x

# =========================================================================
#  模型注册: 使用原始名称，但工厂函数创建新模型
#  这样 main_finetune.py 无需任何改动
# =========================================================================
def Spiking_vit_MetaFormer_Spike_SepConv(depths=[2, 2, 6, 2], **kwargs):
    model = SpikeMambaFormer(
        depths=depths,
        **kwargs,
    )
    return model

def Efficient_Spiking_Transformer_l(**kwargs):
    if 'nb_classes' in kwargs:
        num_classes = kwargs.pop('nb_classes')
    else:
        num_classes = 10
        
    model = SpikeMambaFormer(
        in_channels=3,
        num_classes=num_classes,
        embed_dim=[64, 128, 256, 360],
        depths=[2, 2, 6, 2],
        mlp_ratios=[4, 4, 4, 4],
        **kwargs,
    )
    return model

def Efficient_Spiking_Transformer_m(**kwargs):
    if 'nb_classes' in kwargs:
        num_classes = kwargs.pop('nb_classes')
    else:
        num_classes = 10
        
    model = SpikeMambaFormer(
        in_channels=3,
        num_classes=num_classes,
        embed_dim=[48, 96, 192, 240],
        depths=[2, 2, 6, 2],
        mlp_ratios=[4, 4, 4, 4],
        **kwargs,
    )
    return model

def Efficient_Spiking_Transformer_s(**kwargs):
    if 'nb_classes' in kwargs:
        num_classes = kwargs.pop('nb_classes')
    else:
        num_classes = 10
        
    model = SpikeMambaFormer(
        in_channels=3,
        num_classes=num_classes,
        embed_dim=[32, 64, 128, 192],
        depths=[2, 2, 6, 2],
        mlp_ratios=[4, 4, 4, 4],
        **kwargs,
    )
    return model

def Efficient_Spiking_Transformer_t(**kwargs):
    if 'nb_classes' in kwargs:
        num_classes = kwargs.pop('nb_classes')
    else:
        num_classes = 10
        
    model = SpikeMambaFormer(
        in_channels=3,
        num_classes=num_classes,
        embed_dim=[24, 48, 96, 128],
        depths=[2, 2, 6, 2],
        mlp_ratios=[4, 4, 4, 4],
        **kwargs,
    )
    return model


from timm.models import create_model

if __name__ == "__main__":
    # 测试我们的新模型
    model = Efficient_Spiking_Transformer_s() # 这会创建 SpikeMambaFormer
    print(model)
    x = torch.randn(1, 3, 224, 224)
    y = model(x)
    print("Output shape:", y.shape)
    torchinfo.summary(model, (1, 3, 224, 224), device='cpu')