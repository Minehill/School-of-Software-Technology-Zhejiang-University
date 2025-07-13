# models_hierarchical_mamba.py (修正版)

import torch
import torch.nn as nn
from functools import partial

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model

from mamba_ssm.modules.mamba_simple import Mamba


try:
    from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

class Block(nn.Module):
    def __init__(
        self, dim, mixer_cls, norm_cls=nn.LayerNorm, fused_add_norm=False, residual_in_fp32=False,drop_path=0.,
    ):
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.mixer = mixer_cls(dim)
        self.norm = norm_cls(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"

    def forward(self, hidden_states, residual=None):
        if not self.fused_add_norm:
            if residual is None:
                residual = hidden_states
            else:
                residual = residual + self.drop_path(hidden_states)
            
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            if residual is None:
                hidden_states, residual = fused_add_norm_fn(
                    hidden_states, self.norm.weight, self.norm.bias, residual=residual, prenorm=True,
                    residual_in_fp32=self.residual_in_fp32, eps=self.norm.eps,
                )
            else:
                hidden_states, residual = fused_add_norm_fn(
                    self.drop_path(hidden_states), self.norm.weight, self.norm.bias, residual=residual, prenorm=True,
                    residual_in_fp32=self.residual_in_fp32, eps=self.norm.eps,
                )    
        
        # 双向扫描逻辑
        x_f = hidden_states
        x_b = hidden_states.flip([1])
        
        x_f = self.mixer(x_f)
        x_b = self.mixer(x_b)
        
        hidden_states = x_f + x_b.flip([1])
        
        return hidden_states, residual

def create_block(
    d_model, ssm_cfg=None, norm_epsilon=1e-5, drop_path=0., rms_norm=False,
    residual_in_fp32=False, fused_add_norm=False, layer_idx=None,
    device=None, dtype=None,
):
    if ssm_cfg is None:
        ssm_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}
    
    mixer_cls = partial(Mamba, layer_idx=layer_idx, d_state=16, **ssm_cfg, **factory_kwargs)
    norm_cls = partial(nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs)
    block = Block(
        d_model, mixer_cls, norm_cls=norm_cls, drop_path=drop_path,
        fused_add_norm=fused_add_norm, residual_in_fp32=residual_in_fp32,
    )
    block.layer_idx = layer_idx
    return block

# ... (PatchEmbed, PatchMerging, BasicLayer 类的定义与之前相同，无需修改) ...
class PatchEmbed(nn.Module):
    """ Image to Patch Embedding """
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}x{W}) doesn't match model ({self.img_size[0]}x{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        return x

class PatchMerging(nn.Module):
    """ Patch Merging Layer. """
    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}x{W}) are not even."
        x = x.view(B, H, W, C)
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)
        x = x.view(B, -1, 4 * C)
        x = self.norm(x)
        x = self.reduction(x)
        return x

class BasicLayer(nn.Module):
    """ A basic Vim layer for one stage. """
    def __init__(self, dim, input_resolution, depth, ssm_cfg=None, norm_epsilon=1e-5, rms_norm=False,
                 residual_in_fp32=False, fused_add_norm=False, drop_path=0., downsample=None):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.blocks = nn.ModuleList([
            create_block(
                d_model=dim, ssm_cfg=ssm_cfg, norm_epsilon=norm_epsilon, rms_norm=rms_norm,
                residual_in_fp32=residual_in_fp32, fused_add_norm=fused_add_norm, layer_idx=i,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
            ) for i in range(depth)])
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=nn.LayerNorm)
        else:
            self.downsample = None

    def forward(self, x):
        residual = None
        for blk in self.blocks:
            x, residual = blk(x, residual)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

# ----------------- 多尺度Vim -----------------

class HierarchicalVim(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=[2, 2, 9, 2],
                 ssm_cfg=None, rms_norm=True, residual_in_fp32=True,
                 fused_add_norm=True, drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, 
                 **kwargs):
        super().__init__()

        # print("HierarchicalVim kwargs:", kwargs)

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer)
        
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               ssm_cfg=ssm_cfg, rms_norm=rms_norm,
                               residual_in_fp32=residual_in_fp32, fused_add_norm=fused_add_norm,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                               )
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return set()

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

    # 修改 forward_features 方法，使其接受但忽略不需要的参数
    def forward_features(self, x, **kwargs):
        x = self.patch_embed(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        x = self.avgpool(x.transpose(1, 2))  # B C 1
        x = torch.flatten(x, 1)
        return x

    # 修改 forward 方法，使其签名与原始 VisionMamba 兼容
    def forward(self, x, return_features=False, **kwargs):
        # 调用 forward_features 时，将所有额外的关键字参数传递过去
        features = self.forward_features(x, **kwargs)
        if return_features:
            return features
        
        # head 不接受这些额外参数，所以这里只传递 features
        logits = self.head(features)
        return logits



@register_model
def hier_vim_tiny(pretrained=False, **kwargs):
    model = HierarchicalVim(
        embed_dim=96,
        depths=[2, 2, 9, 2],
        **kwargs
    )
    return model

@register_model
def hier_vim_small(pretrained=False, **kwargs):
    model = HierarchicalVim(
        embed_dim=96,
        depths=[2, 2, 27, 2],
        **kwargs
    )
    return model

@register_model
def hier_vim_base(pretrained=False, **kwargs):
    model = HierarchicalVim(
        embed_dim=128,
        depths=[2, 2, 27, 2],
        **kwargs
    )
    return model