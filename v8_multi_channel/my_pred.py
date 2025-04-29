from ultralytics import YOLO
import datetime
import time
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

###########yolov8多通道推理##########
if __name__ == '__main__':
    model = YOLO(r"D:\Shao\Files\appdata\PycharmProject\v8\v8_ch\v8_multi_channel\runs\detect\train20\weights\best.pt")
    path = r"D:\Shao\Files\appdata\PycharmProject\v8\v8_ch\datasets_my_multi\test.txt"
    model.predict(source=path, ch=8, save=True, conf=0.5, save_txt=True, save_conf=True)


