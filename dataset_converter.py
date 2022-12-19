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

    if("class" in attributes.keys()):
        class_=attributes["class"]
        status = attributes["status"]
        cur_class = class_+'_'+status
        if(cur_class in class_list):
            return cur_class
    # elif ("damagetype" in attributes.keys()):
    #     key = "damagetype"
    # elif("damageType" in attributes.keys()):
    #     key = "damageType"
    # if(key != None):
    #     class_ = attributes[key]
    #     print(class_)
    #     print(json_object["annotations"][anno_index],"여기")



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
        "RoadSafetySign_Normal": 1,
        "RoadSafetySign_Repair": 2,
        "RoadSafetySign_Substitution/Disposal": 3,
        "WalkAcrossPreventionFacility_Normal": 4,
        "WalkAcrossPreventionFacility_Repair": 5,
        "WalkAcrossPreventionFacility_Substitution/Disposal": 6,
        "ProtectionFence_Normal": 7,
        "ProtectionFence_Repair": 8,
        "ProtectionFence_Substitution/Disposal": 9,
        "SignalPole_Normal": 10,
        "SignalPole_Repair": 11,
        "SignalPole_Substitution/Disposal": 12
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
        # "categories": [
        #     {
        #         "id": 1,
        #         "name": "RoadSafetySign_Normal",
        #         "supercategory": "None"
        #     },
        #     {
        #         "id": 2,
        #         "name": "RoadSafetySign_Repair",
        #         "supercategory": "None"
        #     },
        #     {
        #         "id": 3,
        #         "name": "RoadSafetySign_Substitution/Disposal",
        #         "supercategory": "None"
        #     },
        #     {
        #         "id": 4,
        #         "name": "RoadSafetySign_damage",
        #         "supercategory": "None"
        #     },
        #     {
        #         "id": 5,
        #         "name": "RoadSafetySign_discoloration",
        #         "supercategory": "None"
        #     },
        #     {
        #         "id": 6,
        #         "name": "RoadSafetySign_surfacePeeling",
        #         "supercategory": "None"
        #     },
        #     {
        #         "id": 7,
        #         "name": "RoadSafetySign_distortion",
        #         "supercategory": "None"
        #     },
        #     {
        #         "id": 8,
        #         "name": "WalkAcrossPreventionFacility_Normal",
        #         "supercategory": "None"
        #     },
        #     {
        #         "id": 9,
        #         "name": "WalkAcrossPreventionFacility_Repair",
        #         "supercategory": "None"
        #     },
        #     {
        #         "id": 10,
        #         "name": "WalkAcrossPreventionFacility_Substitution/Disposal",
        #         "supercategory": "None"
        #     },
        #     {
        #         "id": 11,
        #         "name": "WalkAcrossPreventionFacility_damage",
        #         "supercategory": "None"
        #     },
        #     {
        #         "id": 12,
        #         "name": "WalkAcrossPreventionFacility_discoloration",
        #         "supercategory": "None"
        #     },
        #     {
        #         "id": 13,
        #         "name": "WalkAcrossPreventionFacility_surfacePeeling",
        #         "supercategory": "None"
        #     },
        #     {
        #         "id": 14,
        #         "name": "WalkAcrossPreventionFacility_distortion",
        #         "supercategory": "None"
        #     },
        #     {
        #         "id": 15,
        #         "name": "ProtectionFence_Normal",
        #         "supercategory": "None"
        #     },
        #     {
        #         "id": 16,
        #         "name": "ProtectionFence_Repair",
        #         "supercategory": "None"
        #     },
        #     {
        #         "id": 17,
        #         "name": "ProtectionFence_Substitution/Disposal",
        #         "supercategory": "None"
        #     },
        #     {
        #         "id": 18,
        #         "name": "ProtectionFence_damage",
        #         "supercategory": "None"
        #     },
        #     {
        #         "id": 19,
        #         "name": "ProtectionFence_discoloration",
        #         "supercategory": "None"
        #     },
        #     {
        #         "id": 20,
        #         "name": "ProtectionFence_surfacePeeling",
        #         "supercategory": "None"
        #     },
        #     {
        #         "id": 21,
        #         "name": "ProtectionFence_distortion",
        #         "supercategory": "None"
        #     },
        #     {
        #         "id": 22,
        #         "name": "SignalPole_Normal",
        #         "supercategory": "None"
        #     },
        #     {
        #         "id": 23,
        #         "name": "SignalPole_Repair",
        #         "supercategory": "None"
        #     },
        #     {
        #         "id": 24,
        #         "name": "SignalPole_Substitution/Disposal",
        #         "supercategory": "None"
        #     },
        #     {
        #         "id": 25,
        #         "name": "SignalPole_damage",
        #         "supercategory": "None"
        #     },
        #     {
        #         "id": 26,
        #         "name": "SignalPole_discoloration",
        #         "supercategory": "None"
        #     },
        #     {
        #         "id": 27,
        #         "name": "SignalPole_surfacePeeling",
        #         "supercategory": "None"
        #     },
        #     {
        #         "id": 28,
        #         "name": "SignalPole_distortion",
        #         "supercategory": "None"
        #     }
        "categories": [
            {
                "id": 1,
                "name": "RoadSafetySign_Normal",
                "supercategory": "None"
            },
            {
                "id": 2,
                "name": "RoadSafetySign_Repair",
                "supercategory": "None"
            },
            {
                "id": 3,
                "name": "RoadSafetySign_Substitution/Disposal",
                "supercategory": "None"
            },
            {
                "id": 4,
                "name": "WalkAcrossPreventionFacility_Normal",
                "supercategory": "None"
            },
            {
                "id": 5,
                "name": "WalkAcrossPreventionFacility_Repair",
                "supercategory": "None"
            },
            {
                "id": 6,
                "name": "WalkAcrossPreventionFacility_Substitution/Disposal",
                "supercategory": "None"
            },
            {
                "id": 7,
                "name": "ProtectionFence_Normal",
                "supercategory": "None"
            },
            {
                "id": 8,
                "name": "ProtectionFence_Repair",
                "supercategory": "None"
            },
            {
                "id": 9,
                "name": "ProtectionFence_Substitution/Disposal",
                "supercategory": "None"
            },
            {
                "id": 10,
                "name": "SignalPole_Normal",
                "supercategory": "None"
            },
            {
                "id": 11,
                "name": "SignalPole_Repair",
                "supercategory": "None"
            },
            {
                "id": 12,
                "name": "SignalPole_Substitution/Disposal",
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
    # json_path = '/home/dgdgksj/facility/datasets/facility/val_label'
    # image_path = '/home/dgdgksj/facility/datasets/facility/val'
    json_path = '/home/dgdgksj/facility/datasets/facility/train_label'
    image_path = '/home/dgdgksj/facility/datasets/facility/train'
    save_path = '/home/dgdgksj/facility/datasets/facility/annotations'


    json_path_list = os.listdir(json_path)
    image_path_list = os.listdir(image_path)
    dataset_category_list = ["RoadSafetySign", "WalkAcrossPreventionFacility", "ProtectionFence", "SignalPole",
                             "damage", "discoloration", "surfacePeeling", "distortion"]
    class_list = ["RoadSafetySign_Normal",
                  "RoadSafetySign_Repair",
                  "RoadSafetySign_Substitution/Disposal",
                  # "RoadSafetySign_damage",
                  # "RoadSafetySign_discoloration",
                  # "RoadSafetySign_surfacePeeling",
                  # "RoadSafetySign_distortion",

                  "WalkAcrossPreventionFacility_Normal",
                  "WalkAcrossPreventionFacility_Repair",
                  "WalkAcrossPreventionFacility_Substitution/Disposal",
                  # "WalkAcrossPreventionFacility_damage",
                  # "WalkAcrossPreventionFacility_discoloration",
                  # "WalkAcrossPreventionFacility_surfacePeeling",
                  # "WalkAcrossPreventionFacility_distortion",

                  "ProtectionFence_Normal",
                  "ProtectionFence_Repair",
                  "ProtectionFence_Substitution/Disposal",
                  # "ProtectionFence_damage",
                  # "ProtectionFence_discoloration",
                  # "ProtectionFence_surfacePeeling",
                  # "ProtectionFence_distortion",

                  "SignalPole_Normal",
                  "SignalPole_Repair",
                  "SignalPole_Substitution/Disposal"
                  # "SignalPole_damage",
                  # "SignalPole_discoloration",
                  # "SignalPole_surfacePeeling",
                  # "SignalPole_distortion"
                  ]

    base_dict = get_base_dict()
    img_cnt = 0
    anno_cnt = 0
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
            base_dict['images'].append(get_image_dict(i, width, height, img_name)[0])
            # print()
            annotations = json_object['annotations']
            # if(len(json_object['annotations'])==1):
            #     continue
                # print(len(json_object['annotations']))
            #     print(json_object['annotations'])
            #     break
            # print('*' * 50)
            # print(annotations)

            for anno_index, data in enumerate(json_object['annotations']):
                # print(data)
                bbox = data['bbox']
                # print(bbox)
                category = get_category(data['attributes'],class_list,json_object,anno_index)
                if(category != None and category in class_list):
                    # print(category, "여기요!")
                    anno_cnt+=1
                    base_dict['annotations'].append(get_annotations_dict(anno_cnt, i, category, bbox)[0])
                # print(bbox)
            # break

                # category, bbox = get_category_bbox(data, dataset_category_list)
            # break
    #         for data in json_object['annotations']:
    #             category,bbox = get_category_bbox(data,category_list)
    #
    #             if(category != None):
    #                 base_dict['annotations'].append(get_annotations_dict(cnt,i,category,bbox)[0])
    #             else:
    #                 pass
    #             cnt+=1
    #
    with open(os.path.join(save_path, './instances_train.json'), 'w') as f:
    # with open(os.path.join(save_path,'./instances_val.json'), 'w') as f:
        json.dump(base_dict, f)
