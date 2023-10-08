# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 15:06:55 2021

@author: admin
"""
import torch
import cv2
import sys
sys.path.append("D:\\deepsort\\ShuffleNet_deepsort\\deep")
import torchvision
import onnxruntime as rt
import numpy as np

from ShuffleNetV2 import shufflenet_v2_x0_5


#modelPath = "./models/Z75/allDataMobileNetV3_.pth"
modelPath = "./best.pth"
#model = get_mobilenetv3(num_classes=2)
#model = get_ShuffleNet_pre(False, num_classes=2, model_path = " ")
#model = get_ShuffleNetV3(False, num_classes=2, model_path = " ")
model = shufflenet_v2_x0_5(num_classes=751)
model.load_state_dict(torch.load(modelPath))
model.eval()
#define resnet18 model
#define input shape
#x = torch.rand(1, 3, 320, 150)
x = torch.rand(1, 3, 64, 128)
#define input and output nodes, can be customized
input_names = ["x"]
output_names = ["y"]
#convert pytorch to onnx
torch_out = torch.onnx.export(model, x, "best.onnx", input_names=input_names, output_names=output_names)
#torch_out = torch.onnx._export(model, x, "kmobilenetv3.onnx", export_params=True)

