import os
import cv2
import sys

from tqdm import tqdm
# json_path = '/home/dgdgksj/facility/datasets/facility/val_label'
# image_path = '/home/dgdgksj/facility/datasets/facility/val'
json_path = '/home/dgdgksj/facility/datasets/facility/train_label'
image_path = '/home/dgdgksj/facility/datasets/facility/train'
# for i in tqdm(range(len(os.listdir(image_path)))):
# sys.stdout = open('error_val.txt', 'w')
image_path_list = os.listdir(image_path)
json_path_list = os.listdir(json_path)
index_list = [458, 596, 2014, 2956, 5286, 6444, 6810, 7254, 7267, 7533, 8275, 9346, 9939, 10574, 11013, 11072, 11103, 12120, 13340, 14941, 15364, 15636, 16014, 16427, 16710, 16896, 18712, 19448, 19470, 21094, 21755, 21927, 22838, 23950, 24234, 24378, 24964, 25203, 27144, 27779, 30678, 30817, 30829, 32451, 32562, 32926, 33278, 33607, 35766, 35939, 36216, 36717, 38333, 38490, 40755, 44827, 46124, 46282, 47605, 48374, 51747, 52074, 52595, 52776, 52781, 53321, 53644, 54125, 54813, 57390, 58638, 58819, 59348, 60732, 61961, 62056, 62488, 64415, 64809, 65254]
cnt = 0
for i in tqdm(range(len(os.listdir(json_path)))):
# for i in range(len(os.listdir(json_path))):
#     i=index_list[cnt]
#     i = 16000+cnt
    i = 24342
    file_name = json_path_list[i]
    img_name = file_name.split('.')[0] + '.jpeg'
    img = cv2.imread(os.path.join(image_path, img_name))
    # print(img_name,cnt,index_list[cnt])
    print(img_name)
    cnt += 1
    # if(cnt ==len(image_path_list)-1):
    #     break
    break
# for i in range(len(os.listdir(image_path))):
#
#     file_name = image_path_list[i]
#     # img_name = file_name.split('.')[0] + '.jpeg'
#     # img = cv2.imread(os.path.join(image_path, image_path_list[i]))
#     img = cv2.imread(os.path.join(image_path, "702_10_0cd1f6f2-f124-4d54-84ce-85019c911018.jpg"))
#     # cv2.imwrite(os.path.join(image_path, "702_10_0cd1f6f2-f124-4d54-84ce-85019c911018.jpg"),img)
#     print(type(img))
#     print(i,image_path_list[i],img.shape)
#
#     # sys.stdout.close()
#     break

#
#
# import os
# from tqdm import tqdm
#
#
# if __name__ == '__main__':
#     img_dir_path = '/home/dgdgksj/facility/datasets/facility/train'
#     json_dir_path = '/home/dgdgksj/facility/datasets/facility/train_label'
#     # # json_path =
#     # # image_path =
#     img_list = os.listdir(img_dir_path)
#     json_list = os.listdir(json_dir_path)
#
#     print(len(img_list))
#     print(len(json_list))
#
#     print(img_list[0] in json_list)
#     print(img_list[0].split('.')[0]+'.json' in json_list)
#
#     for i in tqdm(range(len(img_list))):
#         if(img_list[i].split('.')[0] + '.json' in json_list):
#             pass
#         else:
#             print(img_list[i].split('.')[0])
#
#     for i in tqdm(range(len(json_list))):
#         if(json_list[i].split('.')[0] + '.jpeg' in img_list or json_list[i].split('.')[0] + '.jpg' in img_list):
#             pass
#         else:
#             print(json_list[i].split('.')[0])
#
#
#
