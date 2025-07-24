# 将yolo测试之后的txt文件转换成labelme的json文件

import os
import cv2
import json
import base64
from tqdm import tqdm

# label_dict = {
#     0: "bangzi",
#     1: "zhuantou",
#     2: "chuizi"
# }
label_dict = {
    0: "gun",
    1: "gun",
    2: "gun"
}

def encode_base64(file):
    with open(file,'rb') as f:
        img_data = f.read()
        base64_data = base64.b64encode(img_data)
        # 如果想要在浏览器上访问base64格式图片，需要在前面加上：data:image/jpeg;base64,
        base64_str = str(base64_data, 'utf-8')
    return base64_str

def make_labelme_txt(data_dict):
    for img_path in tqdm(data_dict):
        json_path = img_path.replace('.jpg','.json')
        box_info = data_dict[img_path]
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
            label_id, x_ctr, y_ctr, box_w, box_h = box
            x1 = (x_ctr - box_w / 2)*img_w
            x2 = (x_ctr + box_w / 2)*img_w
            y1 = (y_ctr - box_h / 2)*img_h
            y2 = (y_ctr + box_h / 2)*img_h
            box_data = {
                "label": label_dict[label_id],
                "points": [[x1, y1], [x2, y2]],
                "group_id": idx,
                "description": "",
                "shape_type": "rectangle",
                "flags": {},
            }
            data["shapes"].append(box_data)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)


def read_yolo_gt(dirname):
    data = {}
    for root,dirs,files in os.walk(dirname):
        for file in files:
            if file.endswith(".jpg"):
                image_info = []
                image_path = os.path.join(root, file)
                label_path = image_path.replace("/images/","/labels/").replace(".jpg",".txt")
                if os.path.exists(label_path):
                    with open(label_path, 'r') as f:
                        for line in f:
                            if len(line.strip().split()) == 5:
                                obj_id, x_ctr,y_ctr, box_w, box_h = map(float, line.strip().split())
                                image_info.append([int(obj_id), x_ctr,y_ctr, box_w, box_h])
                data[image_path] = image_info
    return data

def filter_error_yolo(data):
    # 将标注有问题的图片过滤出来
    error_imgs = []
    for img_path in tqdm(data):
        error_flag = False
        box_info = data[img_path]
        for box in box_info:
            label_id, x_ctr, y_ctr, box_w, box_h = box
            # if label_id > 0:
            #     error_flag = True
            if x_ctr < 0 or x_ctr > 1 or y_ctr < 0 or y_ctr > 1 or box_w < 0 or box_w > 1 or box_h < 0 or box_h > 1:
                error_flag = True
        if error_flag:
            # print(box_info)
            # print(img_path)
            error_imgs.append(img_path)
    print("有问题的图片数量: {}".format(len(error_imgs)))
    return error_imgs

def modify_json(data):
    # 修改json文件,将1，2改为0，不符合要求的坐标都去掉
    # 将标注有问题的图片过滤出来
    cnt = 0
    for img_path in tqdm(data):
        need_modify = False
        label_path = img_path.replace("/images/","/labels/").replace(".jpg",".txt")
        box_info = data[img_path]
        new_box_info = []
        for box in box_info:
            label_id, x_ctr, y_ctr, box_w, box_h = box
            if label_id > 0:
                print(box)
                need_modify = True
                need_label_id = 0
            if x_ctr < 0 or x_ctr > 1 or y_ctr < 0 or y_ctr > 1 or box_w < 0 or box_w > 1 or box_h < 0 or box_h > 1:
                print(box)
                continue
            if need_modify:
                new_box_info.append([need_label_id, x_ctr, y_ctr, box_w, box_h])
        # print(len(box_info), len(new_box_info))
        if len(new_box_info) == 0:
            continue
        else:
            cnt += 1
            # with open(label_path, "w") as f:
            #     for box in new_box_info:
            #         f.write(" ".join(map(str, box)) + "\n")
    print("需要修改的label文件数量:", cnt)

if __name__ == '__main__':
    # dirname = "/Work/wangjing/data/SecureATM/data_train_fmt20241106"
    dirname = "/Work/wangjing/data/gun_det_data"
    data = read_yolo_gt(dirname)
    # make_labelme_txt(data)

    # error_imgs = filter_error_yolo(data)
    # with open("error_imgs.txt", "w") as f:
    #     for img_path in error_imgs:
    #         f.write(img_path + "\n")

    modify_json(data)