import json
import cv2
import numpy as np
import os
from tqdm import tqdm

def get_category_bbox(json_object,category_list):
    bbox = json_object['bbox']
    # print(json_object)
    category = None
    if('damagetype' in json_object['attributes']):
        category = json_object['attributes']['damagetype']
        if(category in category_list):
            pass
        else:
            category = None
    elif('damageType' in json_object['attributes']):
        category = json_object['attributes']['damageType']
        if (category in category_list):
            pass
        else:
            category = None
    elif('class' in json_object['attributes']):
        # print("여긴가?")
        category = json_object['attributes']['class']
        # print(category)
        if (category in category_list):
            pass
        else:
            category = None
    return category,bbox

    # return class_

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
        "RoadSafetySign": 1,
        "WalkAcrossPreventionFacility": 2,
        "ProtectionFence": 3,
        "SignalPole": 4,
        "damage": 5,
        "discoloration": 6,
        "surfacePeeling": 7,
        "distortion": 8
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
            "year": 2020,
            "contributor": "",
            "data_created": "2020-12-09"
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
                "name": "RoadSafetySign",
                "supercategory": "None"
            },
            {
                "id": 2,
                "name": "WalkAcrossPreventionFacility",
                "supercategory": "None"
            },
            {
                "id": 3,
                "name": "ProtectionFence",
                "supercategory": "None"
            },
            {
                "id": 4,
                "name": "SignalPole",
                "supercategory": "None"
            },
            {
                "id": 5,
                "name": "damage",
                "supercategory": "None"
            },
            {
                "id": 6,
                "name": "discoloration",
                "supercategory": "None"
            },
            {
                "id": 7,
                "name": "surfacePeeling",
                "supercategory": "None"
            },
            {
                "id": 8,
                "name": "distortion",
                "supercategory": "None"
            },
        ],
        "images": [
        ],
        "annotations": [
        ],
    }
    return dict(base_dict)

if __name__ == '__main__':



    # print(base_dict["licenses"])

    #
    # json_path = '/home/dgdgksj/facility/datasets/facility/train_label'
    # image_path = '/home/dgdgksj/facility/datasets/facility/train'
    json_path = '/home/dgdgksj/facility/datasets/facility/val_label'
    image_path = '/home/dgdgksj/facility/datasets/facility/val'
    save_path = '/home/dgdgksj/facility/datasets/facility/annotations'

    # print(os.listdir(labels_path))
    json_path_list = os.listdir(json_path)
    image_path_list = os.listdir(image_path)
    category_list = ["RoadSafetySign", "WalkAcrossPreventionFacility", "ProtectionFence", "SignalPole", 'damage',
                     "discoloration", "surfacePeeling", "distortion"]
    # test_list =[]
    # for i in range(len(os.listdir(json_path))):
    #     print(image_path_list[i],json_path_list[i])
        # file_name = image_path_list[i]
        # test_list.append(file_name.split('.')[1])
    # print(set(test_list))
        # if(file_name.split('.')[1]=='jpg'):
            # print(os.path.join(image_path,file_name),'->',os.path.join(image_path,file_name.split('.')[0]+'.jpeg'))
            # os.rename(os.path.join(image_path,file_name),os.path.join(image_path,file_name.split('.')[0]+'.jpeg'))


    # for i in tqdm(range(len(os.listdir(json_path)))):
    base_dict = get_base_dict()
    cnt = 0
    for i in tqdm(range(len(os.listdir(json_path)))):
    # for i in range(len(os.listdir(json_path))):
        file_name = json_path_list[i]
        img_name = file_name.split('.')[0]+'.jpeg'
        img = cv2.imread(os.path.join(image_path,img_name))
        height, width, _ = img.shape
        # image_dict = get_image_dict(i,width,height,img_name)
        # print('image dict')
        # print(image_dict,type(image_dict))
        # print('-------------')
        #
        # print('base_dict')
        # print(type(base_dict['images']),base_dict['images'])
        base_dict['images'].append(get_image_dict(i, width, height, img_name)[0])
        # print(base_dict)
        # print('-------------')
        with open(os.path.join(json_path, file_name)) as f:
            json_object = json.load(f)
            cate_bbox = []
            # print(json_object)
            for data in json_object['annotations']:
                # print('\t',data)
                # get_categpry_bbox(data,category_list)
                category,bbox = get_category_bbox(data,category_list)
                # print("category,bbox",category,bbox)
                if(category != None):
                    base_dict['annotations'].append(get_annotations_dict(cnt,i,category,bbox)[0])
                else:
                    pass
                cnt+=1

        # print('\t', cate_bbox)
        # break
        # if(i>30):
        #     break
        # print(base_dict)
        # print('*'*50)
    # print(base_dict)
    # with open(os.path.join(save_path, './instances_train.json'), 'w') as f:
    with open(os.path.join(save_path,'./instances_val.json'), 'w') as f:
        json.dump(base_dict, f)
