import glob
import os
import numpy as np
import sys

current_dir = "/home/dgdgksj/facility/datasets/facility_yolo2/original"
split_pct = 10;

file_train = open("/home/dgdgksj/facility/datasets/facility_yolo2/train.txt", "w")
file_val = open("/home/dgdgksj/facility/datasets/facility_yolo2/val.txt", "w")
counter = 1
index_test = round(100 / split_pct)
for pathAndFilename in glob.iglob(os.path.join(current_dir, "*.jpeg")):
        title, ext = os.path.splitext(os.path.basename(pathAndFilename))
        if counter == index_test:
                counter = 1
                file_val.write(current_dir + "/" + title + '.jpeg' + "\n")
        else:
                file_train.write(current_dir + "/" + title + '.jpeg' + "\n")
                counter = counter + 1
file_train.close()
file_val.close()