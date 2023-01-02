import json
import os
import cv2
from tqdm import tqdm
import shutil

def get_category(attributes):

    key = None

    if ("damagetype" in attributes.keys()):
        key = "damagetype"
    elif("damageType" in attributes.keys()):
        key = "damageType"
    if(key != None):
        class_ = attributes[key]
        return class_
    else:
        return None
def convert(size, box):
    dw = 1./size[0]
    dh = 1./size[1]
    # print(dw,dh)
    x = (box[0] + box[1])/2.0
    y = (box[2] + box[3])/2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    # print(x,y,w,h)
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)
def coco_to_yolo(bbox,image_w,image_h):
    xmin, ymin, w, h = bbox
    b = (xmin,xmin+w,ymin,ymin+h)
    # print(b)
    # print(image_h,image_w)
    # print(convert((image_w,image_h),b))
    return convert((image_w,image_h),b)


def operation_on_a_damaged_type(image_dir_path,json_dir_path,save_path):
    json_path_list = os.listdir(json_dir_path)
    image_path_list = os.listdir(image_dir_path)
    dataset_category_list = ["RoadSafetySign", "WalkAcrossPreventionFacility", "ProtectionFence", "SignalPole",
                             "damage", "discoloration", "surfacePeeling", "distortion"]
    class_list = [
        "damage",
        # "RoadSafetySign_discoloration",
        "surfacePeeling",
        "distortion", ]
    check = False
    cnt = 1
    for i in tqdm(range(len(os.listdir(json_dir_path)))):
        # if(i<9482):
        #     continue
        file_name = json_path_list[i]
        img_name = file_name.split('.')[0] + '.jpeg'
        # print()
        # print(file_name)
        with open(os.path.join(json_dir_path, file_name)) as f:
            json_object = json.load(f)
            # print(json_object)
            width = json_object['images'][0]['width']
            height = json_object['images'][0]['height']
            annotations = json_object['annotations']
            yolo_format_str = ''
            for anno_index, data in enumerate(json_object['annotations']):
                # print(data)
                bbox = data['bbox']
                # print(bbox)
                category = get_category(data['attributes'])
                if (category != None and category in class_list):
                    # print(category, "여기요!")

                    for index, class_ in enumerate(class_list):
                        if(category == class_):
                            # print(bbox)
                            x, y, w, h = coco_to_yolo(bbox=bbox, image_w=width, image_h=height)


                            yolo_format_str += str(index) + ' ' + str(x) + ' ' + str(y) + ' ' + str(w) + ' ' + str(
                                h) + ' ' + '\n'
                            if (w > 1 or y > 1 or w > 1 or h > 1):
                                check=True
                                # print(i,yolo_format_str)
                            # if(cnt==4797):
                            #     print(yolo_format_str)
                            break

            if(yolo_format_str!='' and check==False):
                # print(yolo_format_str)
                shutil.copy(os.path.join(image_dir_path,img_name),os.path.join(save_path,str(cnt).zfill(6))+'.jpeg')
                f = open(os.path.join(save_path,str(cnt).zfill(6))+'.txt', 'w')
                f.write(yolo_format_str)
                f.close()
                cnt+=1
            else:
                check=False
                # if(cnt>3):
                #     break



if __name__ == '__main__':
    image_dir_path='/home/dgdgksj/facility/datasets/facility_yolo2/train'
    json_dir_path = '/home/dgdgksj/facility/datasets/facility_yolo2/train_label'
    save_dir_path = '/home/dgdgksj/facility/datasets/facility_yolo2/original'
    operation_on_a_damaged_type(image_dir_path=image_dir_path, json_dir_path=json_dir_path,save_path=save_dir_path)

