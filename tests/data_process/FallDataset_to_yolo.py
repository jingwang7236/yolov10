# 解析Le2i Fall Dataset 数据集中的视频和Box信息，转换成yolo格式的标注文件
import os
import cv2

def parse_label(label_file):
    data = {}
    lines = open(label_file).read().splitlines()[2:]
    for line in lines:
        if len(line.split(",")) != 6:
            continue
        frame_id, label, box_x1, box_y1, box_x2, box_y2 = map(int,line.split(","))
        if frame_id not in data:
            data[frame_id] = [[box_x1, box_y1, box_x2, box_y2]]
        else:
            data[frame_id].append([box_x1, box_y1, box_x2, box_y2])
    return data


def main(dirpath, rootdir):
    video_path = os.path.join(dirpath, "Videos")
    label_path = os.path.join(dirpath, "Annotation_files")
    image_path = os.path.join(rootdir, "images")
    txt_path = os.path.join(rootdir, "labels")
    for root,dirnames,filenames in os.walk(video_path):
        for filename in filenames:
            if filename.endswith(".avi"):
                video_file = os.path.join(root, filename)
                label_file = os.path.join(label_path, filename.replace(".avi", ".txt"))
                video_name = video_file.split("/")[-1].split(".")[0].replace("(","").replace(")","")
                batch_name = dirpath.split("/")[-1]
                print(video_file)
                print(label_file)
                try:
                    data = parse_label(label_file)
                except:
                    print(filename)
                    import pdb; pdb.set_trace()
                # 读取视频
                video = cv2.VideoCapture(video_file)
                width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
                while True:
                    ret, frame = video.read()
                    if not ret:
                        break
                    
                    frame_id = int(video.get(cv2.CAP_PROP_POS_FRAMES))
                    if frame_id % 20 == 0 and frame_id  in data:  # 每隔20帧保存一张图片
                        annot_lst = []
                        for obj in data[frame_id]:
                            xmin, ymin, xmax, ymax = obj
                            x_ctr = (xmin + xmax)//2/width
                            y_ctr = (ymin + ymax)//2/height
                            box_w = (xmax - xmin)/width
                            box_h = (ymax - ymin)/height
                            annot = [0, x_ctr,y_ctr, box_w, box_h]
                            annot_lst.append(annot)
                        # cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    # cv2.imshow("frame", frame)
                    # cv2.waitKey(1)
                        if len(annot_lst) == 0:
                            continue
                        image_name = "{}_{}_{}.jpg".format(batch_name, video_name, frame_id)
                        # 保存图片
                        cv2.imwrite(os.path.join(image_path, image_name), frame)
                        with open(os.path.join(txt_path, image_name.replace(".jpg",".txt")),'w') as f:
                            for obj in annot_lst:
                                f.write(' '.join(map(str,obj))+'\n')


if __name__ == '__main__':
    rootdir = "/Work/wangjing/data/person_det_data/PublicData/FallDataset"
    dirnames = ["Home_01", "Home_02", "Coffee_room_01", "Coffee_room_02"]
    for dirname in dirnames:
        dirpath = os.path.join(rootdir, dirname)
        main(dirpath, rootdir)