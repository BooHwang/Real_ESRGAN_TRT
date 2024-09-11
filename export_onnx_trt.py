#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File      : export_onnx_trt.py
@Time      : 2024/09/10 15:46:20
@Author    : Huang Bo
@Contact   : cenahwang0304@gmail.com
@Desc      : None
'''

'''
/root/TensorRT-8.5.1.7/targets/x86_64-linux-gnu/bin/trtexec --onnx=realesr.opt.onnx --minShapes=img:1x3x256x256 --optShapes=img:3x3x256x256 --maxShapes=img:5x3x256x256 --fp16 --saveEngine=realesr-general_x4v3_fp16_bs1-5-5.plan
'''

import os
import cv2
import onnx
import time
import torch
import numpy as np
import onnxruntime
import torch.nn as nn
import torch.nn.functional as F
from onnxsim import simplify
from pprint import pprint
from basicsr.archs.srvgg_arch import SRVGGNetCompact

import warnings
warnings.filterwarnings("ignore")


def check_model(onnx_path, onnx_opt_path):
    model = onnx.load(onnx_path)
    simplified_model, check = simplify(model)
    assert check, "Simplified ONNX model could not be validated"
    onnx.save(simplified_model, onnx_opt_path)
    try:
        onnx.checker.check_model(onnx_opt_path)
        print("model check passed!")
    except Exception as e:
        print(f"model check failed: {e}")
        
def print_in_out_name(onnx_opt_path):
    onnx_model = onnx.load(onnx_opt_path)
    onnx_session = onnxruntime.InferenceSession(onnx_model.SerializeToString())

    print("------input-------")
    input_tensors = onnx_session.get_inputs()
    for input_tensor in input_tensors:
        input_info = {
            "name": input_tensor.name,
            "type": input_tensor.type,
            "shape": input_tensor.shape,
        }
        pprint(input_info)
        
    print("------output-------")
    output_tensors = onnx_session.get_outputs()
    for output_tensor in output_tensors:
        output_info = {
            "name" : output_tensor.name,
            "type" : output_tensor.type,
            "shape": output_tensor.shape,
        }
        pprint(output_info)


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
        
    @torch.no_grad()
    def forward(self, img):
        h = img.shape[2]
        img = img / 255.
        img = img.half()
        
        output_img = self.model(img)
        
        # in face, model can x4 super resolution, if you want, can commit this line
        output_img = F.interpolate(output_img, size=(h*2, h*2), mode='bilinear', align_corners=False)
        output_img = output_img.float().clamp_(0, 1)
        output_img *= 255.0
        output_img = output_img.to(torch.uint8)
        
        return output_img

if __name__ == "__main__":
    net = REALESGAN()
    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

    img = cv2.imread("data/96.png")
    img = img[:, :, ::-1]
    img = torch.from_numpy(img.copy())
    img = img.permute(2, 0, 1)
    img = img.unsqueeze(0).to(device)
    print(img.shape, img.dtype)
    
    # test model inference
    output = net(img)
    print(output.shape)
    cv2.imwrite("x4.png", output.squeeze().permute(1, 2, 0).cpu().numpy()[..., ::-1])

    # export to onnx
    onnx_path = 'realesr.onnx'
    onnx_opt_path = onnx_path.replace(".onnx", ".opt.onnx")
    torch.onnx.export(
        net,
        img,
        onnx_path,
        opset_version=17,
        input_names=['img'],
        output_names=['output'],
        dynamic_axes={"img": {0: "bs", 2: "h", 3: "w"},
                    "output": {0: "bs", 2: "h", 3: "w"},
                    },
    )
    check_model(onnx_path, onnx_opt_path)
    print_in_out_name(onnx_opt_path)

    
