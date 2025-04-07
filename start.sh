from ultralytics import YOLOv10

# header
model = YOLOv10()
model.train(data='bank_header_voc.yaml', epochs=100, batch=256, imgsz=640,
            device='5,6', pretrained=False, )
model.val(data='bank_voc.yaml', batch=256)



# If you want to finetune the model with pretrained weights, you could load the 
# pretrained weights like below
# model = YOLOv10.from_pretrained('jameslahm/yolov10{n/s/m/b/l/x}')
# or
# wget https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10{n/s/m/b/l/x}.pt
# model = YOLOv10('yolov10{n/s/m/b/l/x}.pt')

# train the model---可以跑通,epoch=100,map不再提升
# (yolov10) wangjing7236@hxrj-X640-G40:/Work/wangjing/code/yolov10-main$
# yolo detect train data=VOC.yaml model=yolov10n/s/m/b/l/x.yaml epochs=500 batch=256 imgsz=640 device=0,1,2,3,4,5,6,7
# yolo detect train data=bank_voc.yaml model=bank_yolov10n.yaml epochs=100 batch=256 imgsz=640 device=4,5
# yolo val model=runs/detect/train15/weights/best.pt data=bank_voc.yaml batch=1 imgsz=640



# # End-to-End ONNX
# yolo export model=jameslahm/yolov10{n/s/m/b/l/x} format=onnx opset=13 simplify
# yolo export model=runs/detect/train/weights/best.pt format=onnx opset=13 simplify

# # Predict with ONNX
# yolo predict model=yolov10n/s/m/b/l/x.onnx

# # End-to-End TensorRT
# yolo export model=jameslahm/yolov10{n/s/m/b/l/x} format=engine half=True simplify opset=13 workspace=16
# # or
# trtexec --onnx=yolov10n/s/m/b/l/x.onnx --saveEngine=yolov10n/s/m/b/l/x.engine --fp16
# # Predict with TensorRT
# yolo predict model=yolov10n/s/m/b/l/x.engine
