from ultralytics import YOLO

# 모델 로드하기
model = YOLO('yolov8n.pt')  # 공식 모델을 로드합니다.
model = YOLO('/mnt/home/jo/Facility_Damage_Detection/Camera_Detection/runs/detect_prevention_diagonal/train3/weights/best.pt')  # 사용자 정의 모델을 로드합니다.

# 모델 검증하기
metrics = model.val()  # 데이터셋과 설정을 기억하니 인수는 필요 없습니다.
metrics.box.map    # map50-95
metrics.box.map50  # map50
metrics.box.map75  # map75
metrics.box.maps   # 각 카테고리의 map50-95가 포함된 리스트입니다.