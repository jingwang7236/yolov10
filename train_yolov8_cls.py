
# from ultralytics import YOLOv10
from ultralytics import YOLO

# header
model = YOLO(model='ultralytics/cfg/models/v8/bank-yolov8-hat-cls.yaml') # 从头开始构建新模型
# model = YOLOv10(model='yolov10n.pt')  # 加载预训练模型

model.train(data='/Work/wangjing/data/face_attr_data/yolo_data/yolo_hat_dataset/',
            epochs=100, batch=480, imgsz=224,device='6', 
            pretrained=False, multi_scale=True, project='runs/yolov8-cls/hat')

# yolo val model=runs/yolov8-cls/hat/train/weights/best.pt data=/Work/wangjing/data/face_attr_data/yolo_data/yolo_hat_dataset batch=16 imgsz=224 project=runs/yolov8-cls/hat

# train --训练集
# test --验证集,在训练过程中测试的验证集
# val --测试集，在训练结束后的测试集