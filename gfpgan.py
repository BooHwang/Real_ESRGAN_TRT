#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File      : gfpgan.py
@Time      : 2024/09/10 09:43:02
@Author    : Huang Bo
@Contact   : cenahwang0304@gmail.com
@Desc      : None
'''

'''
borrow from: https://huggingface.co/spaces/Xintao/GFPGAN
'''

import os
import cv2
import time
import torch
from basicsr.archs.srvgg_arch import SRVGGNetCompact
from realesrgan.utils import RealESRGANer
from gfpgan.utils import GFPGANer

import warnings
warnings.filterwarnings("ignore")


img = cv2.imread("data/96.png")
device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

# super resolution
model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
model_path = 'weights/realesr-general-x4v3.pth'
half = True if torch.cuda.is_available() else False
upsampler = RealESRGANer(scale=4, model_path=model_path, model=model, tile=0, tile_pad=10, pre_pad=0, half=half, device=device)

output, img_mode = upsampler.enhance(img)
cv2.imwrite("x1.png", output)

img_resized = cv2.resize(output, (512, 512), interpolation=cv2.INTER_LINEAR)
cv2.imwrite("x2.png", img_resized)

# face enhance
face_enhancer = GFPGANer(model_path='GFPGANv1.4.pth', upscale=2, arch='clean', channel_multiplier=2, bg_upsampler=upsampler, device=device) # upsampler

t0 = time.time()
_, _, output = face_enhancer.enhance(img, has_aligned=False, only_center_face=False, paste_back=True, weight=0.5)
print(f"enhance use time: {(time.time()-t0):.3f} s")
cv2.imwrite("x3.png", output)