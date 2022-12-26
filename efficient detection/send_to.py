import os

import shutil
from tqdm import tqdm
filename = 'test.txt'
# src = '/home/banana/'
# dir = '/home/banana/txt/'
# shutil.move(src + filename, dir + filename)



if __name__ == '__main__':
    json_path = '/home/dgdgksj/facility/datasets/labels'
    image_path = '/home/dgdgksj/facility/datasets/images'

    train_image_path = '/home/dgdgksj/facility/datasets/facility/train'
    val_image_path = '/home/dgdgksj/facility/datasets/facility/val'
    train_json_path = '/home/dgdgksj/facility/datasets/facility/train_label'
    val_json_path = '/home/dgdgksj/facility/datasets/facility/val_label'

    # print(os.listdir(labels_path))
    json_path_list = os.listdir(json_path)
    image_path_list = os.listdir(image_path)

    data_len = len(image_path_list)
    train_ = data_len//10 * 7
    val = data_len - train_
    # print(train)
    # print(val)
    # print(train+val,data_len)
    #
    # for i, data in enumerate(json_path_list):
    for i in tqdm(range(len(json_path_list))):
        # print(json_path_list[i].split('.')[0])
        # print(image_path_list[i].split('.')[0])
        filename = image_path_list[i].split('.')[0]
        img_filename = filename + ".jpeg"
        json_filename = filename + ".json"
        src_image_path = os.path.join(image_path,img_filename)
        src_json_path = os.path.join(json_path, json_filename)
        # print(src_image_path)
        # print(src_json_path)
        # break
        if(i<train_):
            dst_image_path = os.path.join(train_image_path,img_filename)
            dst_json_path = os.path.join(train_json_path, json_filename)
            shutil.move(src_image_path, dst_image_path)
            # print('*'*50)
            # print(src_json_path,dst_json_path)
            # print('*' * 50)
            shutil.move(src_json_path, dst_json_path)
        else:
            dst_image_path = os.path.join(val_image_path, img_filename)
            dst_json_path = os.path.join(val_json_path, json_filename)
            shutil.move(src_image_path, dst_image_path)
            shutil.move(src_json_path, dst_json_path)
        # print(dst_image_path)
        # print(dst_json_path)
        # break
        # os.path.join(train_path,img_filename)