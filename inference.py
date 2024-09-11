#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File      : inference.py
@Time      : 2024/09/11 15:19:36
@Author    : Huang Bo
@Contact   : cenahwang0304@gmail.com
@Desc      : None
'''

'''
pytorch版本的超分推理
'''

import cv2
import torch
import torch.nn as nn
import numpy as np
from basicsr.archs.srvgg_arch import SRVGGNetCompact


class REALESGAN(nn.Module):
    def __init__(self, model_path='weights/realesr-general-x4v3.pth', gpu_id=0):
        super().__init__()
        self.device = torch.device(f"cuda:{str(gpu_id)}" if torch.cuda.is_available() else "cpu")

        self.model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
        self.half = True if torch.cuda.is_available() else False
        state_dict = torch.load(model_path, map_location="cpu")
        self.model.load_state_dict(state_dict["params"], strict=True)
        self.model.eval()
        self.model = self.model.to(self.device)
        self.model = self.model.half()
        
    def pre_process(self, img):
        """Pre-process, such as pre-pad and mod pad, so that the images can be divisible
        """
        img = torch.from_numpy(np.transpose(img, (2, 0, 1))).float()
        self.img = img.unsqueeze(0).to(self.device)
        if self.half:
            self.img = self.img.half()
        
    @torch.no_grad()
    def __call__(self, img):
        img = img.astype(np.float32)
        img = img / 255
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        self.pre_process(img)
        output_img = self.model(self.img)
        print(output_img.shape)
        output_img = output_img.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        output_img = np.transpose(output_img[[2, 1, 0], :, :], (1, 2, 0))
        output = (output_img * 255.0).round().astype(np.uint8)

        return output
    
    
if __name__ == "__main__":
    net = REALESGAN()
        
    img = cv2.imread("data/96.png")

    output = net(img)
    print(output.shape)

    cv2.imwrite("x4.png", output)