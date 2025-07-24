
# from ultralytics import YOLOv10
from ultralytics import YOLO

# header
model = YOLO(model='bank_yolov8.yaml') # 从头开始构建新模型
# model = YOLO(model='yolov8n.pt')  # 加载预训练模型
# model.train(data='bank_header_voc.yaml', model='bank_yolov10n.yaml', 
#             epochs=100, batch=256, imgsz=640,device='5,6', 
#             pretrained=False, project='runs/header')

# 开启多尺度训练，导致单批次内显存需求波动，需要降低batch,不然会导致显存不足
# 使用iou训练需要更多epoch，才能让预测边界框尽可能回归到真实目标边界框中
# model.train(data='bank_person_voc.yaml', model='bank_yolov8.yaml', 
#             epochs=500, batch=96, imgsz=640,device='6', 
#             pretrained=False, multi_scale=True, project='runs/yolov8/person')

# model.train(data='bank_person_head.yaml', model='bank_yolov8.yaml', 
#             epochs=500, batch=128, imgsz=640,device='6,7', 
#             pretrained=False, multi_scale=True, project='runs/yolov8/person_head')

# model.train(data='bank_header_voc.yaml', model='bank_yolov8.yaml', 
#             epochs=200, batch=128, imgsz=640,device='4,6', 
#             pretrained=True, multi_scale=True, project='runs/yolov8/header')

# model.train(data='bank_screen_voc.yaml', model='bank_yolov8.yaml', 
#             epochs=200, batch=96, imgsz=640,device='4', 
#             pretrained=False, multi_scale=True, project='runs/yolov8/screen')

# model.train(data='bank_knife_det.yaml', model='bank_yolov8.yaml', 
#             epochs=100, batch=96, imgsz=640,device='3', 
#             pretrained=False, multi_scale=True, project='runs/yolov8/knife')

model.train(data='bank_gun_det.yaml', model='bank_yolov8.yaml', 
            epochs=100, batch=96, imgsz=640,device='1', 
            pretrained=False, multi_scale=True, project='runs/yolov8/gun')

# 混合精度训练，设置amp=True,可以降低显存占用 
# 但是需要谨慎使用multi_scale=True,不同尺寸会导致loss差异较大,

# model.train(data='bank_person_voc.yaml', model='bank_yolov10n.yaml', 
#             epochs=200, batch=256, imgsz=640,device='5,6', 
#             pretrained=False, project='runs/person')

# 测试的时候使用model.val没有预测结果，使用yolo val可以正常预测
# model.val(data='bank_header_voc.yaml', batch=16, imgsz=640, 
#           project='runs/header',
#           model='runs/header/train/weights/best.pt')

# yolo detect train data=bank_voc.yaml model=bank_yolov10n.yaml epochs=100 batch=256 imgsz=640 device=4,5
# yolo val model=runs/detect/train15/weights/best.pt data=bank_header_voc.yaml batch=16 imgsz=640 project=runs/header
# yolo val model=runs/yolov8/screen/train4/weights/best.pt data=bank_screen_voc.yaml batch=16 imgsz=640 project=runs/yolov8/screen
# yolo val model=runs/yolov8L/person/train17/weights/best.pt data=bank_person_voc.yaml batch=16 imgsz=640 project=runs/yolov8L/person save_txt=True save_conf=True

# export onnx，适用于算能平台，ultralytics/nn/tasks.py _predict_once函数的return x修改为：return x.permute(0, 2, 1)

# model = YOLO(model='runs/yolov8/person/train10/weights/best.pt')  # 加载模型
# model = YOLO(model='yolov8s.pt')
# model.export(format='onnx', opset=17, dynamic=True)
