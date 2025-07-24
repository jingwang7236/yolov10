# voc训练集列表转换为yolo训练集格式
import os
import cv2
import xml.etree.ElementTree as ET
from tqdm import tqdm

# root_dir = '/Work/wangjing/data/person_det_data'
# train_list = [
#     # 'PublicData/CrowdHuman/train_person.txt', # 密集行人公开数据集
#     # 'PublicData/CrowdHuman/val_person.txt', # 密集行人公开数据集
#     # 'WiderPerson/train_person.txt', 
#     # 'GuiZhouBankImages/batch_01_det_all.txt',  # 贵州银行场景5k张
#     # 'GuiZhouBankImages/batch_02_det_all.txt',  # 贵州银行场景5k张
#     # 'GuiZhouBankImages/batch_03_det_all.txt',  # 贵州银行场景5k张
#     # 'GuiZhouBankImages/batch_04_det_all.txt',  # 贵州银行场景5k张
#     # 'GuiZhouBankImages/batch_05_det_all.txt',  # 贵州银行场景5k张
#     # 'GuiZhouBankImages/batch_06_det_all.txt',  # 贵州银行场景5k张
#     'GuiZhouBankImages/batch_07_det_all.txt',  # 贵州银行场景5k张中挑出hardcase
#     'GuiZhouBankImages/batch_08_det_all.txt',  # 贵州银行场景5k张中挑出hardcase
#     'GuiZhouBankImages/batch_11_det_all.txt',  # 贵州银行场景5k张中挑出hardcase
#     # 'TestSet/ATM_det_all.txt',
#     # 'TestSet/支行管理端测试视频0108_det_all.txt',  # 测试找的银行视频，有每个业务场景的视频
#     # 'TestSet/guizhou_show_202502_det_all.txt',  # 贵州银行poc演示数据，包含所有场景
# ]
# tagnames = {
#     'person': 0,
#     # 'ignore_region': -2,  # 忽略mask
#     'hard_person': 0,  # 人体遮挡严重
#     # 'head': 1,
#     # 'header': 1,
# }
root_dir = '/Work/wangjing/data/phone_screen_data'
train_list = [
    'yixing_data_det_all.txt',
    'guizhou_data_2024_det_all.txt',
    'company_show_data_det_all.txt',
    'guizhou_data_2025_det_all.txt',
    'guizhou_poc_show_02.txt'
]

tagnames = {
    'phone': 0,
    'screen':1,
    'hard_phone': 0,  # 黑屏手机加入训练-玩手机不用亮屏控制
}

# root_dir = '/Work/wangjing/data/header_det_data'
# train_list = [
#     #'ShuangLuData/dataset_2018_det_all.txt',  # 双录项目提供的数据demo，后续应该作为测试集
#     'ShuangLuData/spider_dataset_20250429_det_all.txt',  # 爬虫双录相似数据
#     ]
# tagnames = {
#     'head': 0,
#     'header': 0,
# }

def read_xml(xml_path, img_path):
    tree = ET.parse(xml_path)
    target = tree.getroot()
    width = float(target.find('size').find('width').text)
    height = float(target.find('size').find('height').text)
    if width == 0 or height == 0:
        height, width, _ = cv2.imread(img_path).shape
    data = []
    try:
        for obj in target.iter('object'):
            name = obj.find('name').text
            if name not in tagnames:
                continue
            obj_id = tagnames[name]
            bbox = obj.find('bndbox')
            x0 = float(bbox.find('xmin').text)
            y0 = float(bbox.find('ymin').text)
            x1 = float(bbox.find('xmax').text)
            y1 = float(bbox.find('ymax').text)
            xmin = min(x0,x1)
            xmax = max(x0,x1)
            ymin = min(y0,y1)
            ymax = max(y0,y1)
            # annot = [xmin,ymin,xmax,ymax,name]
            x_ctr = (xmin + xmax)//2/width
            y_ctr = (ymin + ymax)//2/height
            box_w = (xmax - xmin)/width
            box_h = (ymax - ymin)/height
            annot = [obj_id, x_ctr,y_ctr, box_w, box_h]
            data.append(annot)
    except:
        import pdb;pdb.set_trace()
    return data

def convert_voc2yolo(file_path, label_dirname):
    with open(file_path, 'r') as f:
        for line in tqdm(f):
            img_path, xml_path = line.strip().split()
            img_path = os.path.join(root_dir, img_path)
            xml_path = os.path.join(root_dir, xml_path)
            # if "/images/" in xml_path:
            #     txt_path = xml_path.replace('.xml', '.txt').replace('/images/', '/{}/'.format(label_dirname))
            if "/images/" in img_path:
                txt_path = img_path.replace('.jpg', '.txt').replace('/images/', '/{}/'.format(label_dirname))
            else:
                print("目录结构不符合yolo规范")
                import pdb; pdb.set_trace()
            os.makedirs(os.path.dirname(txt_path), exist_ok=True)
            yolo_data = read_xml(xml_path, img_path)
            # 适用于CrowdHuman数据集,将person和header标签合并
            # if "/PersonAnnotations/" in xml_path:
            #     header_xml_path = xml_path.replace('/PersonAnnotations/', '/HeaderAnnotations/')
            #     if os.path.exists(header_xml_path):
            #         header_yolo_data = read_xml(header_xml_path, img_path)
            #         yolo_data.extend(header_yolo_data)
            # if not yolo_data:
            #     continue
            with open(txt_path,'w') as f2:
                for obj in yolo_data:
                    f2.write(' '.join(map(str,obj))+'\n')

if __name__ == '__main__':
    label_dirname = "labels"
    # label_dirname = "person_labels"
    # label_dirname = "person_head_labels"
    for file_name in train_list:
        file_path = os.path.join(root_dir, file_name)
        print(file_path)
        convert_voc2yolo(file_path, label_dirname)
