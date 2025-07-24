# 跟踪demo,输入视频
import sys
import cv2

from ultralytics import YOLO

# Load the YOLOv8n model
# model = YOLO("yolov8n.pt")
model = YOLO("runs/yolov8/person/train7/weights/best.pt")

# Open the video file
# 押运人员进出大门，偶尔会因为检测不到而id增加，导致计数不准确
# video_path = "/Work/wangjing/code/project/dsz1206.mp4"  
video_path = sys.argv[1]
cap = cv2.VideoCapture(video_path)

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        # botsort是DeepSORT的增强版
        results = model.track(frame, persist=True, tracker="bytetrack.yaml")
        # results = model.track(frame, persist=True, tracker="botsort.yaml")
        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Display the annotated frame
        cv2.imshow("YOLOv8 Tracking", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()