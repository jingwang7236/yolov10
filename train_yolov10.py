
from ultralytics import YOLOv10

# header
model = YOLOv10(model='bank_yolov10n.yaml') # 从头开始构建新模型
# model = YOLOv10(model='yolov10n.pt')  # 加载预训练模型
# model.train(data='bank_header_voc.yaml', model='bank_yolov10n.yaml', 
#             epochs=100, batch=256, imgsz=640,device='5,6', 
#             pretrained=False, project='runs/header')

# 开启多尺度训练，导致单批次内显存需求波动，需要降低batch,不然会导致显存不足
model.train(data='bank_person_voc.yaml', model='bank_yolov10n.yaml', 
            epochs=200, batch=128, imgsz=640,device='6,7', 
            pretrained=False, multi_scale=True, project='runs/person')

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
