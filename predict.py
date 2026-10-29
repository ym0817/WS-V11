from ultralytics import YOLO

# 加载训练好的模型，改为自己的路径
model = YOLO(r'D:\BaiduNetdiskDownload\v10_Ink\WS-YOLO-main\runs\train\exp5\weights\best.pt')

# source = r'D:\BaiduNetdiskDownload\v10_Ink\ultralytics-main\frame_9_0038_0_2\images_no_objects'

# source = r'D:\BaiduNetdiskDownload\v10_Ink\ultralytics-main\frame_9_0038_0_2\images_objects'

source = r'D:\BaiduNetdiskDownload\v10_Ink\YOLO26_RD_main\Data_HSMD\images'  # 修改为自己的图片路径及文件名

# 运行推理，并附加参数
# model.predict(source, )
model.predict(source,  # 图片或视频的源目录
              conf=0.38,  # 用于检测的 对象置信阈值，只有置信度高于此阈值的对象才会被检测出来
              iou=0.45,  # 非极大值抑制(NMS)的交并比(loU)值
              imgsz=640,  # 输入图像尺寸
              # half=False,  # 使用半精度(FP16)
              device=0,  # 运行设备，如device=0或device = cpu
              # max_det=300,  # 每个图像的最大检测数
              # vid_srtide=False,#视频帧率步长
              stream_buffer=False,  # 缓冲所有流帧( True )或 返回最近的帧( Fa1se )
              visualize=False,  # 是否可视化模型特征
              augment=False,  # 是否对预测源应用图像增强
              agnostic_nms=False,  # 是否使用类别不可知(无关)的非极大值抑制(NMS)
              classes=None,  # 按类别筛选结果，即classes=0或classes=[0,2,3]
              retina_masks=False,  # 是否使用高分辨率的分割掩膜
              embed=None,  # 返回给定层的特征向量/嵌入
              show=False,  # 如果环境允许，是否显示预测的图像和视频
              save_frames=False,  # 是否保存预测的单个视频帧
              save_txt=False,  # 是否将结果保存为 .txt 文件
              save_crop=False,  # 是否保存裁剪的图像与结果
              show_labels=True,  # 是否显示预测标签
              show_conf=True,  # 是否显示预测置信度
              show_boxes=True,  # 是否显示预测边界框
              line_width=None,  # 边界框的线宽(如果为 None ，则缩放为图像大小)
              save=True
              )