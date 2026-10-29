# coding=utf-8
import torch
from ultralytics import YOLO


# model_weight = r"C:\EaglesLib\TensorRT8616\bin\best.pt"
model_weight = r"D:\BaiduNetdiskDownload\v10_Ink\WS-YOLO-main\runs\train\exp5\weights\best.pt"
# model_weight = r"D:\BaiduNetdiskDownload\v10_Ink\ultralytics-main\backup_detect_HS160\weights\best.pt"
# model_weight = r'D:\BaiduNetdiskDownload\v10_Ink\ultralytics-main\Data_smalldie\weights\best.pt'
model = YOLO(model_weight)
# metrics = model.val()
# path = model.export(format="onnx",imgsz = 640,opset=17)
# model.export(format="engine")

onnx_path = model.export(
    format="onnx",
    imgsz=640,
    opset=17,
    # dynamic=True,
    # half=True,
    # int8=True,              # 【核心】开启 INT8 量化
)
# quantize=8,        # 新参数，启用 INT8 量化
