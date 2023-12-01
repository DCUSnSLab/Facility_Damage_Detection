from ultralytics import YOLO
from setproctitle import *

setproctitle('camera detection Train_WalkAcrossPreventionFacilitys')
model = YOLO('yolov8n.yaml')
model = YOLO('yolov8n.pt')
model = YOLO('yolov8n.yaml').load('yolov8n.pt')

results = model.train(data = '/mnt/home/jo/Facility_Damage_Detection/Dataset/yaml/Traffic_Facility.yaml', epochs=100, imgsz=640)
