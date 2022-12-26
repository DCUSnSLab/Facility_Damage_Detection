import json
import cv2
import numpy as np
import os
from tqdm import tqdm


def get_category(attributes, class_list, json_object,anno_index):
    # print(attributes)
    check = False
    cur_class = None
    key = None

    # if("class" in attributes.keys()):
    #     class_=attributes["class"]
    #     status = attributes["status"]
    #     cur_class = class_+'_'+status
    #     if(cur_class in class_list):
    #         return cur_class
    if ("damagetype" in attributes.keys()):
        key = "damagetype"
    elif("damageType" in attributes.keys()):
        key = "damageType"
    if(key != None):
        class_ = attributes[key]
        return class_
        # print(class_)
        # print(json_object["annotations"][anno_index],"여기")
    else:
        return None

def get_image_dict(id,width,height,file_name):
    base_dict = {
                    "id": id,
                    "file_name": file_name,
                    "width": width,
                    "height": height,
                    "date_captured": "",
                    "license": 1,
                    "coco_url": "",
                    "flickr_url": ""
                },
    return base_dict
def get_annotations_dict(id,image_id,category,bbox):
    cate = {
        "damage": 1,
        "surfacePeeling": 2,
        "distortion": 3,
    }

    x, y, width, height = bbox
    base_dict = {
                    "id": id,
                    "image_id": image_id,
                    "category_id": cate[category],
                    "iscrowd": 0,
                    "area": height * width,
                    "bbox": [x, y, width, height],
                    "segmentation": []
                },
    return base_dict

def get_base_dict():
    base_dict = {
        "info": {
            "description": "",
            "url": "",
            "version": "",
            "year": 2022,
            "contributor": "",
            "data_created": "2022-12-15"
        },
        "licenses": [
            {
                "id": 1,
                "name": None,
                "url": None
            }
        ],
        "categories": [
            {
                "id": 1,
                "name": "damage",
                "supercategory": "None"
            },
            {
                "id": 2,
                "name": "surfacePeeling",
                "supercategory": "None"
            },
            {
                "id": 3,
                "name": "distortion",
                "supercategory": "None"
            }
        ],
        "images": [
        ],
        "annotations": [
        ],
    }
    return dict(base_dict)

if __name__ == '__main__':
    json_path = '/home/dgdgksj/facility/datasets/facility/val_label'
    image_path = '/home/dgdgksj/facility/datasets/facility/val'
    # json_path = '/home/dgdgksj/facility/datasets/facility/train_label'
    # image_path = '/home/dgdgksj/facility/datasets/facility/train'
    save_path = '/home/dgdgksj/facility/datasets/facility/annotations'


    json_path_list = os.listdir(json_path)
    image_path_list = os.listdir(image_path)
    dataset_category_list = ["RoadSafetySign", "WalkAcrossPreventionFacility", "ProtectionFence", "SignalPole",
                             "damage", "discoloration", "surfacePeeling", "distortion"]
    class_list = [
        "damage",
        # "RoadSafetySign_discoloration",
        "surfacePeeling",
        "distortion",]

    base_dict = get_base_dict()
    img_cnt = 0
    anno_cnt = 0
    check = False
    for i in tqdm(range(len(os.listdir(json_path)))):
    # for i in range(len(os.listdir(json_path))):
        file_name = json_path_list[i]
        img_name = file_name.split('.')[0]+'.jpeg'
        # img = cv2.imread(os.path.join(image_path,img_name))
        # height, width, _ = img.shape
        with open(os.path.join(json_path, file_name)) as f:
            json_object = json.load(f)
            # print(json_object)
            width = json_object['images'][0]['width']
            height = json_object['images'][0]['height']

            annotations = json_object['annotations']

            for anno_index, data in enumerate(json_object['annotations']):
                # print(data)
                bbox = data['bbox']
                # print(bbox)
                category = get_category(data['attributes'],class_list,json_object,anno_index)
                if(category != None and category in class_list):
                    # print(category, "여기요!")
                    check = True
                    anno_cnt+=1
                    base_dict['annotations'].append(get_annotations_dict(anno_cnt, img_cnt, category, bbox)[0])
            if(check):
                check = False
                base_dict['images'].append(get_image_dict(img_cnt, width, height, img_name)[0])
                img_cnt+=1

    # with open(os.path.join(save_path, './instances_train.json'), 'w') as f:
    with open(os.path.join(save_path,'./instances_val.json'), 'w') as f:
        json.dump(base_dict, f)
