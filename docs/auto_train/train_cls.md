# 分类模型训练步骤

## Installation
`conda` virtual environment is recommended. 
```
conda create -n yolov10 python=3.9
conda activate yolov10
pip install -r requirements.txt
pip install -e .
```
## Dataset Preparation
数据集目录中新建每个类的一个子目录,每个子目录都以相应的类命名，并包含该类的所有图像。
每个图像文件都有唯一的命名，并且通常采用常见的图像文件格式，例如 JPEG 或 PNG。

举例如下：

在../ultralytics/datasets新建一个Hat_Data文件夹，子文件夹train、val，每个文件夹下对应类别名称的文件夹，里面放对应的图片。

data 参数，直接对应数据集根目录名称：Hat_Data

```
Hat_Data/
|-- train/
    |-- hat/
    |   |-- img1.jpg
    |   |-- img2.jpg
    |   |-- ...
    |-- no_hat/
    |   |-- img1.jpg
    |   |-- img2.jpg
    |   |-- ...
|-- val/
    |-- hat/
    |   |-- img1.jpg
    |   |-- img2.jpg
    |   |-- ...
    |-- no_hat/
    |   |-- img1.jpg
    |   |-- img2.jpg
    |   |-- ...
```

## Training
```
conda activate yolov10 # 进入环境
(yolov10)$ python train_yolov8_cls.py # 执行训练代码                                             
```

## Evaluation

```
(yolov10)$ yolo val model=runs/yolov8-cls/hat/train/weights/best.pt data=/Work/wangjing/data/face_attr_data/yolo_data/yolo_hat_dataset batch=16 imgsz=224 project=runs/yolov8-cls/hat

```

## Export Model
```
# export onnx model
(yolov10) wangjing7236@hxrj-X640-G40:/Work/wangjing/code/yolov10-main$ yolo export model=runs/yolov8-cls/hat/train/weights/best.pt format=onnx opset=13 simplify

# export int8 rk3588 rknn model

(yolov10) /code/yolov10$ conda activate rknn
(rknn) /code/yolov10$ cd tests/rknn_convert/
(rknn) /code/yolov10/tests/rknn_convert$ python convert.py ../../runs/yolov8-cls/hat/train/weights/best.onnx rk3588

```