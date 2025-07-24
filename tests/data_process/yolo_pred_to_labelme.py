# 将yolo测试之后的txt文件转换成labelme的json文件

import os
import cv2
import json
import base64

def read_yolo_txt(dirname):
    # 返回预测框在0.2-0.5之间的文件名和相应的box
    data = {}
    for root,dirs,files in os.walk(dirname):
        for file in files:
            if file.endswith(".txt"):
                file_path = os.path.join(root, file)
                imgname = file_path.split("/")[-1].split(".")[0]
                badcase = False
                box_info = []
                with open(file_path, "r") as f:
                    for line in f:
                        line = line.strip().split()
                        obj_id, x_ctr,y_ctr, box_w, box_h,conf = map(float, line)
                        if conf >= 0.2:
                            box_info.append([x_ctr,y_ctr, box_w, box_h])
                            if conf < 0.5:
                                badcase = True
                if badcase and box_info:
                    data[imgname] = box_info
    print("符合要求的图片数量：{}".format(len(list(data.keys()))))
    # import pdb;pdb.set_trace()
    return data

def read_list_txt(fname):
    img_dict = {}
    with open(fname, 'r') as f:
        for line in f:
            imgname = line.strip().split("/")[-1].split(".")[0]
            img_dict[imgname] = line.strip()
    return img_dict

def encode_base64(file):
    with open(file,'rb') as f:
        img_data = f.read()
        base64_data = base64.b64encode(img_data)
        # 如果想要在浏览器上访问base64格式图片，需要在前面加上：data:image/jpeg;base64,
        base64_str = str(base64_data, 'utf-8')  
    return base64_str

def make_labelme_txt(data_dict, img_dict):
    for imgname in data_dict:
        img_path = img_dict[imgname]
        json_path = img_path.replace('.jpg','.json')
        box_info = data_dict[imgname]
        img_h,img_w,_ = cv2.imread(img_path).shape
        data = {
            "version": "5.2.1",
            "flags": {},
            "shapes": [],
            "imagePath": img_path.split("/")[-1],
            "imageData": encode_base64(img_path),
            "imageHeight": img_h,
            "imageWidth": img_w
        }
        for idx,box in enumerate(box_info):
            x_ctr, y_ctr, box_w, box_h = box
            x1 = (x_ctr - box_w / 2)*img_w
            x2 = (x_ctr + box_w / 2)*img_w
            y1 = (y_ctr - box_h / 2)*img_h
            y2 = (y_ctr + box_h / 2)*img_h
            box_data = {
                "label": "person",
                "points": [[x1, y1], [x2, y2]],
                "group_id": idx,
                "description": "",
                "shape_type": "rectangle",
                "flags": {},
            }
            data["shapes"].append(box_data)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

if __name__ == '__main__':
    dirname = "runs/yolov8/person/val7/labels/"
    fname = "/Work/wangjing/data/person_det_data/GuiZhouBankImages/银行室内多人场景数据整理.list"
    data = read_yolo_txt(dirname)
    img_dict = read_list_txt(fname)

    make_labelme_txt(data, img_dict)