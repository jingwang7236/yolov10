# coding:utf-8
'''
# 可视化yolo val之后的数据
(yolov10) wangjing7236@hxrj-X640-G40:/Work/wangjing/code/yolov10-main$
 yolo val model=runs/yolov8/person/train11/weights/best.pt data=bank_person_voc.yaml batch=16 imgsz=640 project=runs/yolov8/person save_txt=True save_conf=True
'''

import os
import cv2

def read_imgs(test_dir):
    imgs = []
    for root,dirs,files in os.walk(test_dir):
        for file in files:
            if file.endswith(".jpg"):
                imgs.append(os.path.join(root,file))
    return imgs

def read_txt(val_result, conf=0.5):
    data = {}
    for root,dirs,files in os.walk(val_result):
        for file in files:
            if file.endswith(".txt"):
                key = file.split("/")[-1].split(".")[0]
                filepath = os.path.join(root,file)
                lines = open(filepath).read().splitlines()
                for line in lines:
                    obj_id, x_ctr,y_ctr, box_w, box_h, score = map(float, line.split(" ")) #0 0.698012 0.662356 0.248018 0.638777 0.828706
                    if score >= conf:
                        if key in data:
                            data[key].append([int(obj_id), x_ctr,y_ctr, box_w, box_h, score])
                        else:
                            data[key] = [[int(obj_id), x_ctr,y_ctr, box_w, box_h, score]]
    return data

def draw_img_box(data):
    line_color = (0, 255, 0)
    line_thickness = 2
    line_type = 4
    for imgpath in data:
        image = cv2.imread(imgpath)
        img_h,img_w,_ = image.shape
        lines = data[imgpath]
        for line in lines:
            obj_id, x_ctr,y_ctr, box_w, box_h, score = line
            box_x1 = int((x_ctr-box_w/2) * img_w)
            box_y1 = int((y_ctr-box_h/2) * img_h)
            box_x2 = int((x_ctr+box_w/2) * img_w)
            box_y2 = int((y_ctr+box_h/2) * img_h)
            cv2.rectangle(image, (box_x1,box_y1), (box_x2,box_y2), line_color, line_thickness, line_type)  # 
            cv2.putText(image, str(score) , (box_x1, box_y1-10), cv2.FONT_HERSHEY_PLAIN, 1.0, line_color, line_type)  # 绘字
        new_img_path = imgpath + "_draw_box.jpg"
        cv2.imwrite(new_img_path, image)

def crop_img_box(data):
    for imgpath in data:
        image = cv2.imread(imgpath)
        img_h,img_w,_ = image.shape
        lines = data[imgpath]
        for idx,line in enumerate(lines):
            obj_id, x_ctr,y_ctr, box_w, box_h, score = line
            box_x1 = int((x_ctr-box_w/2) * img_w)
            box_y1 = int((y_ctr-box_h/2) * img_h)
            box_x2 = int((x_ctr+box_w/2) * img_w)
            box_y2 = int((y_ctr+box_h/2) * img_h)
            crop_img_path = "{}_crop_box_{}.jpg".format(imgpath, idx)
            crop_image = image[box_y1:box_y2, box_x1:box_x2]
            cv2.imwrite(crop_img_path, crop_image)

if __name__ == '__main__':
    # test_dir = "/Work/wangjing/data/person_det_data/CompanyShowData/comp_show_fall_person/images"
    # val_result = "runs/yolov8/person/val12/labels"
    test_dir = "/Work/wangjing/data/person_det_data/tmp_data/company_show_data/images"
    val_result = "runs/yolov8/person/val23/labels"
    imgs = read_imgs(test_dir)
    print(len(imgs))
    preds = read_txt(val_result, conf=0.5)
    print(len(preds))
    data = {}
    for imgpath in imgs:
        key = imgpath.split("/")[-1].split(".")[0]
        if key in preds:
            info = preds[key]
            data[imgpath] = info
        else:
            continue
            # import pdb;pdb.set_trace()
    # draw_img_box(data)
    crop_img_box(data)