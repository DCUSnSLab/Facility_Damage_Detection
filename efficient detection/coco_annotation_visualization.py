import os
import cv2
import json

def get_image_path(image_dir_path,image_name):

    if(image_name in os.listdir(image_dir_path)):
        return True
    else:
        return False

def get_image_idx(image_list,image_name):
    for i, data in enumerate(image_list):
        cur_file_name = data['file_name']
        if(image_name == cur_file_name):
            return data['id']
def get_annotations(anno_list,image_idx):
    # anno_list = []
    # cate = {
    #     '1': "RoadSafetySign_Normal",
    #     '2': "RoadSafetySign_Repair",
    #     '3': "RoadSafetySign_Substitution/Disposal",
    #     '4': "WalkAcrossPreventionFacility_Normal",
    #     '5': "WalkAcrossPreventionFacility_Repair",
    #     '6': "WalkAcrossPreventionFacility_Substitution/Disposal",
    #     '7': "ProtectionFence_Normal",
    #     '8': "ProtectionFence_Repair",
    #     '9': "ProtectionFence_Substitution/Disposal",
    #     '10': "SignalPole_Normal",
    #     '11': "SignalPole_Repair",
    #     '12': "SignalPole_Substitution/Disposal"
    # }
    cate = {
        '1': "damage",
        '2': "surfacePeeling",
        '3': "distortion"
    }
    annotations = []
    for i, data in enumerate(anno_list):
        cur_image_idx = data['image_id']
        # print(cur_image_idx)
        if(image_idx == cur_image_idx):
            # print(data.keys())

            #     print(json_object["annotations"][anno_index],"여기")
            category = cate[str(data['category_id'])]
            bbox = data['bbox']
            # print(bbox)
            annotations.append((category,bbox))
    return annotations

if __name__ == '__main__':
    image_dir_path1 = '/home/dgdgksj/facility/datasets/facility/train'
    image_dir_path2 = '/home/dgdgksj/facility/datasets/facility/val'
    json_dir_path = '/home/dgdgksj/facility/datasets/facility/annotations'
    image_name = "802_20_36f83ef5-61b0-4a11-a9be-e2012d2567b7.jpeg"
    check = False
    image_path = None
    json_path = None
    if(get_image_path(image_dir_path=image_dir_path1,image_name=image_name)):
        check = True
        image_path = os.path.join(image_dir_path1,image_name)
        json_path = os.path.join(json_dir_path,'instances_train.json')
    if(get_image_path(image_dir_path=image_dir_path2,image_name=image_name)):
        check = True
        image_path = os.path.join(image_dir_path2, image_name)
        json_path = os.path.join(json_dir_path, 'instances_val.json')
    if(check):
        cv2.namedWindow("img",cv2.WINDOW_NORMAL)
        img=cv2.imread(image_path)

        with open(json_path) as f:
            json_object = json.load(f)
            image_list = json_object['images']
            anno_list = json_object['annotations']
            # print(type(anno_list))
            # print(image_list)
            image_idx = get_image_idx(image_list=image_list,image_name=image_name)
            annotations = get_annotations(anno_list=anno_list,image_idx=image_idx)
        # print(annotations)
        for i in annotations:
            category, bbox = i
            x1,y1,w,h = bbox
            # x1 = int(x1)
            # y1 = int(y1)
            # x2 = int(x1+w)
            # y2 = int(y1+h)
            # x1= 1125.2
            # y1 = 466.1
            # w = 826.0
            # h = 981.4
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x1 + w)
            y2 = int(y1 + h)
            print(x1,y1,x2,y2)
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 0), 5)
            cv2.putText(img, category,
                        (x1, y1 +50), cv2.FONT_HERSHEY_SIMPLEX, 2,
                        (255, 255, 0), 2)
            # print(category,bbox)
        cv2.imshow("img",img)
        cv2.waitKey(30000)
    else:
        assert 1==0,"파일명 확인 해주세요!"



