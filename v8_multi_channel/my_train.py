from ultralytics import YOLO
import datetime
import time
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
# ###########修改初期验证yolov8在rgb图上是正常运行的##########
# if __name__ == '__main__':
#     model = YOLO("yolov8.yaml")
#     model.train(data="datasets_my_multi.yaml", ch=3, epochs=2)
#     metrics = model.val(ch=3)  # evaluate model performance on the validation set
#     metrics = model.val(split="test", ch=3)  # evaluate model performance on the validation set
#

###########yolov8多通道运行##########
if __name__ == '__main__':
    model = YOLO("yolov8.yaml")
    model.train(data="datasets_my_multi.yaml", ch=8, epochs=1)
    metrics = model.val(ch=8)  # evaluate model performance on the validation set
    metrics = model.val(split="test", ch=8)  # evaluate model performance on the validation set


