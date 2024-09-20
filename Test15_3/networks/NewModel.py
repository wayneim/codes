import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from einops import rearrange
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from torchvision.models.resnet import resnet34 as resnet
import numpy as np

import math
from functools import partial


# 新加入的模块
class unetConv2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
               
        return x

class unetUp(nn.Module):
    def __init__(self, in_chan, out_chan, is_up=True):
        super().__init__()
        self.conv = unetConv2(out_chan*2, out_chan)
        self.up = nn.ConvTranspose2d(in_chan, out_chan, kernel_size=4, stride=2, padding=1)
        self.is_up = is_up

    def forward(self, x, skip): 
        if self.is_up == True:
            x = self.up(x)
        x = self.conv(torch.cat([x, skip], dim=1))
        return x
    
  

# 网络
class SwinTransformerSys(nn.Module):
    def __init__(self, img_size=224, in_chans=3, num_classes=9,
                 depths=[3, 4, 6, 3], num_heads=[1, 2, 5, 8],
                 qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super().__init__()

        print("------PVTv2 initialization---")

        self.num_classes = num_classes

        self.num_stages=4
        linear=True
        embed_dims=[64, 128, 320, 512]
        mlp_ratios=[8, 8, 4, 4]
        sr_ratios=[8, 4, 2, 1]

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0

        for i in range(self.num_stages):
            block = nn.ModuleList([Block(
                dim=embed_dims[i], num_heads=num_heads[i], mlp_ratio=mlp_ratios[i], qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + j], norm_layer=norm_layer,
                sr_ratio=sr_ratios[i], linear=linear)
                for j in range(depths[i])])
            norm = norm_layer(embed_dims[i])
            cur += depths[i]
            
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)
        
        # build resnet
        self.resnet=resnet()
        self.drop = nn.Dropout2d(drop_rate)


        # build decoder layers
        # filters = [64, 256, 512, 1024, 2048]
        # filters = [96, 96, 192, 384, 768]
        filters = [64, 64, 128, 320, 512]

        self.up_concat3 = unetUp(filters[4], filters[3])
        self.up_concat2 = unetUp(filters[3], filters[2])
        self.up_concat1 = unetUp(filters[2], filters[1])
        self.up_x4 = nn.ConvTranspose2d(filters[1], filters[0], kernel_size=4, stride=4)
        self.up_x4_norm =nn.BatchNorm2d(filters[0])
        self.segmentationHead = nn.Conv2d(in_channels=filters[0], out_channels=self.num_classes, kernel_size=1)


        self.conv3 = nn.Conv2d(256, 320, kernel_size=1)
        self.conv33 = nn.Conv2d(320, 256, kernel_size=1)

       
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
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}


    #Dencoder and Skip connection
    def forward_features(self, x):
        x_0 = self.resnet.conv1(x)
        x_0 = self.resnet.bn1(x_0)
        x_0 = self.resnet.relu(x_0)
        x_0 = self.resnet.maxpool(x_0)

        # resnet1
        x_1 = self.resnet.layer1(x_0)
        x_1 = self.drop(x_1)

        # pvt1
        block = getattr(self, f"block1")
        norm = getattr(self, f"norm1")
        y_1 = x_1.permute(0, 2, 3, 1).reshape(x_1.shape[0], -1, 64) # [B,L,C]
        for blk in block:
            y_1 = blk(y_1, x_1.shape[2], x_1.shape[3])
        y_1 = norm(y_1)
        y_1 = y_1.permute(0, 2, 1).reshape(x_1.shape[0], -1, x_1.shape[2], x_1.shape[3]) 

        # resnet2
        x_2 = self.resnet.layer2(y_1)
        x_2 = self.drop(x_2)

        # pvt2
        block = getattr(self, f"block2")
        norm = getattr(self, f"norm2")
        y_2 = x_2.permute(0, 2, 3, 1).reshape(x_2.shape[0], -1, 128) # [B,L,C]
        for blk in block:
            y_2 = blk(y_2, x_2.shape[2], x_2.shape[3])
        y_2 = norm(y_2)
        y_2 = y_2.permute(0, 2, 1).reshape(x_2.shape[0], -1, x_2.shape[2], x_2.shape[3]) 


        # resnet3
        x_3 = self.resnet.layer3(y_2)
        x_3 = self.drop(x_3)

        # pvt3
        block = getattr(self, f"block3")
        norm = getattr(self, f"norm3")
        y_3 = self.conv3(x_3).permute(0, 2, 3, 1).reshape(x_3.shape[0], -1, 320) # [B,L,C]
        for blk in block:
            y_3 = blk(y_3, x_3.shape[2], x_3.shape[3])
        y_3 = norm(y_3)
        y_3 = y_3.permute(0, 2, 1).reshape(x_3.shape[0], -1, x_3.shape[2], x_3.shape[3])
        x_4 = self.conv33(y_3) 

        # resnet4
        x_4 = self.resnet.layer4(x_4)
        x_4 = self.drop(x_4)

        # pvt4
        block = getattr(self, f"block4")
        norm = getattr(self, f"norm4")
        y_4 = x_4.permute(0, 2, 3, 1).reshape(x_4.shape[0], -1, 512) # [B,L,C]
        for blk in block:
            y_4 = blk(y_4, x_4.shape[2], x_4.shape[3])
        y_4 = norm(y_4)
        y_4 = y_4.permute(0, 2, 1).reshape(x_4.shape[0], -1, x_4.shape[2], x_4.shape[3])


        x = self.up_concat3(y_4, y_3)
        x = self.up_concat2(x, y_2)
        x = self.up_concat1(x, y_1)
        x = self.up_x4(x)
        x = self.up_x4_norm(x)
        x = self.segmentationHead(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)

        return x


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., linear=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.linear = linear

        if self.linear:
            self.relu = nn.ReLU(inplace=True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        if self.linear:
            x = self.relu(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1, linear=False):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.linear = linear
        self.sr_ratio = sr_ratio
        if not linear:
            if sr_ratio > 1:
                self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
                self.norm = nn.LayerNorm(dim)
        else:
            self.pool = nn.AdaptiveAvgPool2d(7)
            self.sr = nn.Conv2d(dim, dim, kernel_size=1, stride=1)
            self.norm = nn.LayerNorm(dim)
            self.act = nn.GELU()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if not self.linear:
            if self.sr_ratio > 1:
                x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
                x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
                x_ = self.norm(x_)
                kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            else:
                kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(self.pool(x_)).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            x_ = self.act(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):
    
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1, linear=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio, linear=linear)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, linear=linear)
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))

        return x
