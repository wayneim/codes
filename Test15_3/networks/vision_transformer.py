# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging

import torch
import torch.nn as nn
# from .swin_transformer_unet_skip_expand_decoder_sys import SwinTransformerSys
from .NewModel import SwinTransformerSys
logger = logging.getLogger(__name__)

class SwinUnet(nn.Module):
    def __init__(self, img_size=224, num_classes=9):
        super().__init__()

        self.swin_unet = SwinTransformerSys(img_size=img_size, # 根据config.py和yaml参数更新
                                in_chans=3,
                                num_classes=num_classes,
                                depths=[3, 4, 6, 3],
                                num_heads=[1, 2, 5, 8],
                                qkv_bias=True,
                                qk_scale=None,
                                drop_rate=0.0,
                                drop_path_rate=0.1)

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1) # 1维变3维
        logits = self.swin_unet(x)
        return logits


    def load_from_resnet(self, path):
        if path is not None:
            print(f"resnet pretrained_path:{path}")
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print("---start load pretrained modle of resnet encoder---")
            full_dict = torch.load(path, map_location=device)
            full_dict = self.update2(full_dict)

            model_dict = self.swin_unet.state_dict()

            for k in list(full_dict.keys()): # 删除shape不一样的权重
                if k in model_dict:
                    if full_dict[k].shape != model_dict[k].shape:
                        print("delete:{};shape pretrain:{};shape model:{}".format(k,model_dict[k].shape))
                        del full_dict[k]

            msg = self.swin_unet.load_state_dict(full_dict, strict=False)
            print('---resnet state_dict load successfully!---')
    
    def update2(self, state_dict):
        pre = state_dict
        new = {}
        for k, v in pre.items():
            newkey = 'resnet.' + k
            new[newkey] = v
        print('pretrained dict update successfully!')
        return new
    
    def load_from_pvt(self, path):
        if path is not None:
            print(f"pvt pretrained_path:{path}")
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print("---start load pretrained modle of pvt encoder---")
            full_dict = torch.load(path, map_location=device)

            model_dict = self.swin_unet.state_dict()

            for k in list(full_dict.keys()): # 删除shape不一样的权重
                if k in model_dict:
                    if full_dict[k].shape != model_dict[k].shape:
                        print("delete:{};shape pretrain:{};shape model:{}".format(k,model_dict[k].shape))
                        del full_dict[k]

            msg = self.swin_unet.load_state_dict(full_dict, strict=False)
            print('---pvt state_dict load successfully!---')