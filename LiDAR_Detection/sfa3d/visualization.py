from __future__ import print_function
from __future__ import division
import os
import sys
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import math
from torch.utils.data import DataLoader
from builtins import int
from torch.utils.data import Dataset
import open3d

src_dir = os.path.dirname(os.path.realpath(__file__))
while not src_dir.endswith("sfa3d"):
    src_dir = os.path.dirname(src_dir)
if src_dir not in sys.path:
    sys.path.append(src_dir)

from config import test_config as cnf

class bev_utils:
    def makeBEVMap(PointCloud_, boundary):
        Height = cnf.BEV_HEIGHT + 1
        Width = cnf.BEV_WIDTH + 1

        # Discretize Feature Map
        PointCloud = np.copy(PointCloud_)
        # print("PointCloud_Lengh : ", len(PointCloud))
        # PointCloud[:, 0] = np.int_(np.floor(PointCloud[:, 0] / cnf.DISCRETIZATION))
        PointCloud[:, 0] = np.int_(np.floor(PointCloud[:, 0] / cnf.DISCRETIZATION) + Height / 2)
        # PointCloud[:, 0] = np.int_(np.floor(PointCloud[:, 0] / cnf.DISCRETIZATION))
        PointCloud[:, 1] = np.int_(np.floor(PointCloud[:, 1] / cnf.DISCRETIZATION) + Width / 2)

        # sort-3times
        sorted_indices = np.lexsort((-PointCloud[:, 2], PointCloud[:, 1], PointCloud[:, 0]))
        # print("sorted_indices : ", sorted_indices)
        PointCloud = PointCloud[sorted_indices]
        # print("len(PointCloud) : ", len(PointCloud))
        _, unique_indices, unique_counts = np.unique(PointCloud[:, 0:2], axis=0, return_index=True, return_counts=True)
        PointCloud_top = PointCloud[unique_indices]

        # Height Map, Intensity Map & Density Map
        heightMap = np.zeros((Height, Width))
        intensityMap = np.zeros((Height, Width))
        densityMap = np.zeros((Height, Width))

        # some important problem is image coordinate is (y,x), not (x,y)
        max_height = float(np.abs(boundary['maxZ'] - boundary['minZ']))
        heightMap[np.int_(PointCloud_top[:, 0]), np.int_(PointCloud_top[:, 1])] = PointCloud_top[:, 2] / max_height

        normalizedCounts = np.minimum(1.0, np.log(unique_counts + 1) / np.log(64))
        intensityMap[np.int_(PointCloud_top[:, 0]), np.int_(PointCloud_top[:, 1])] = PointCloud_top[:, 3]
        densityMap[np.int_(PointCloud_top[:, 0]), np.int_(PointCloud_top[:, 1])] = normalizedCounts

        RGB_Map = np.zeros((3, Height - 1, Width - 1))
        RGB_Map[2, :, :] = densityMap[:cnf.BEV_HEIGHT, :cnf.BEV_WIDTH]  # r_map
        RGB_Map[1, :, :] = heightMap[:cnf.BEV_HEIGHT, :cnf.BEV_WIDTH]  # g_map
        RGB_Map[0, :, :] = intensityMap[:cnf.BEV_HEIGHT, :cnf.BEV_WIDTH]  # b_map
        return RGB_Map

    # bev image coordinates format
    def get_corners(x, y, w, l, yaw):
        bev_corners = np.zeros((4, 2), dtype=np.float32)
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        # front left
        bev_corners[0, 0] = x - w / 2 * cos_yaw - l / 2 * sin_yaw
        bev_corners[0, 1] = y - w / 2 * sin_yaw + l / 2 * cos_yaw

        # rear left
        bev_corners[1, 0] = x - w / 2 * cos_yaw + l / 2 * sin_yaw
        bev_corners[1, 1] = y - w / 2 * sin_yaw - l / 2 * cos_yaw

        # rear right
        bev_corners[2, 0] = x + w / 2 * cos_yaw + l / 2 * sin_yaw
        bev_corners[2, 1] = y + w / 2 * sin_yaw - l / 2 * cos_yaw

        # front right
        bev_corners[3, 0] = x + w / 2 * cos_yaw - l / 2 * sin_yaw
        bev_corners[3, 1] = y + w / 2 * sin_yaw + l / 2 * cos_yaw

        return bev_corners

    def drawRotatedBox(img, x, y, w, l, yaw, color):
        bev_corners = bev_utils.get_corners(x, y, w, l, yaw)
        corners_int = bev_corners.reshape(-1, 1, 2).astype(int)
        cv2.polylines(img, [corners_int], True, color, 2)
        corners_int = bev_corners.reshape(-1, 2).astype(int)
        cv2.line(img, (corners_int[0, 0], corners_int[0, 1]), (corners_int[3, 0], corners_int[3, 1]), (255, 255, 0), 2)

class dataloader:
    def create_train_dataloader(self):
        """Create dataloader for training"""
        train_lidar_aug = transformation.OneOf([
            transformation.Random_Rotation(limit_angle=np.pi / 4, p=1.0),
            transformation.Random_Scaling(scaling_range=(0.95, 1.05), p=1.0),
        ], p=0.66)
        train_dataset = Dataset(self, mode='train', lidar_aug=train_lidar_aug, hflip_prob=self.hflip_prob,
                                     num_samples=self.num_samples)
        train_sampler = None
        if self.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=(train_sampler is None),
                                      pin_memory=self.pin_memory, num_workers=self.num_workers,
                                      sampler=train_sampler)

        return train_dataloader, train_sampler

    def create_val_dataloader(self):
        """Create dataloader for validation"""
        val_sampler = None
        val_dataset = Dataset(self, mode='val', lidar_aug=None, hflip_prob=0., num_samples=self.num_samples)
        if self.distributed:
            val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
        val_dataloader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False,
                                    pin_memory=self.pin_memory, num_workers=self.num_workers, sampler=val_sampler)
        print(val_dataloader)
        return val_dataloader

    def create_test_dataloader(self):
        """Create dataloader for testing phase"""

        test_dataset = Dataset(self, mode='test', lidar_aug=None, hflip_prob=0.,
                                    num_samples=self.num_samples)
        test_sampler = None
        if self.distributed:
            test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
        test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False,
                                     pin_memory=self.pin_memory, num_workers=self.num_workers,
                                     sampler=test_sampler)

        return test_dataloader

class Dataset(Dataset):
    def __init__(self, configs, mode='test', lidar_aug=None, hflip_prob=None, num_samples=None):
        self.dataset_dir = configs.dataset_dir
        self.input_size = configs.input_size
        self.hm_size = configs.hm_size

        self.num_classes = configs.num_classes
        self.max_objects = configs.max_objects

        assert mode in ['train', 'val', 'test'], 'Invalid mode: {}'.format(mode)
        self.mode = mode
        self.is_test = (self.mode == 'test')
        sub_folder = 'test_dataset' if self.is_test else 'train_dataset'
        # sub_folder = 'train_dataset' if self.is_test else 'test_dataset'
        # sub_folder = 'testing' if self.is_test else 'training'
        self.lidar_aug = lidar_aug
        self.hflip_prob = hflip_prob
        self.image_dir = os.path.join(self.dataset_dir, sub_folder, "image")
        self.lidar_dir = os.path.join(self.dataset_dir, sub_folder, "lidar")
        self.calib_dir = os.path.join(self.dataset_dir, "calib")
        self.label_dir = os.path.join(self.dataset_dir, sub_folder, "label")
        self.sample_id_list = sorted([file_name[:-4] for file_name in os.listdir(self.lidar_dir) if file_name.endswith('.bin')])

        if num_samples is not None:
            self.sample_id_list = self.sample_id_list[:num_samples]
        self.num_samples = len(self.sample_id_list)

    def __len__(self):
        return len(self.sample_id_list)

    def __getitem__(self, index):
        if self.is_test:
            return self.load_img_only(index)
        else:
            return self.load_img_with_targets(index)

    def load_img_only(self, index):
        """Load only image for the testing phase"""
        sample_id = int(self.sample_id_list[index])
        img_path, img_rgb = self.get_image(sample_id)
        lidarData = self.get_lidar(sample_id)
        lidarData = data_utils.get_filtered_lidar(lidarData, cnf.boundary)
        bev_map = bev_utils.makeBEVMap(lidarData, cnf.boundary)
        bev_map = torch.from_numpy(bev_map)

        metadatas = {
            'img_path': img_path,
        }

        return metadatas, bev_map, img_rgb

    def load_img_with_targets(self, index):
        """Load images and targets for the training and validation phase"""
        sample_id = int(self.sample_id_list[index])
        img_path = os.path.join(self.image_dir, '{:06d}.png'.format(sample_id))
        lidarData = self.get_lidar(sample_id)
        # print('lidarData_84 : ', lidarData)
        calib = self.get_calib(sample_id)
        labels, has_labels = self.get_label(sample_id)
        if has_labels:
            labels[:, 1:] = transformation.camera_to_lidar_box(labels[:, 1:], calib.V2C, calib.R0, calib.P2)

        if self.lidar_aug:
            lidarData, labels[:, 1:] = self.lidar_aug(lidarData, labels[:, 1:])

        lidarData, labels = data_utils.get_filtered_lidar(lidarData, cnf.boundary, labels)

        bev_map = bev_utils.makeBEVMap(lidarData, cnf.boundary)
        bev_map = torch.from_numpy(bev_map)

        hflipped = False
        if np.random.random() < self.hflip_prob:
            hflipped = True
            # C, H, W
            bev_map = torch.flip(bev_map, [-1])

        targets = self.build_targets(labels, hflipped)

        metadatas = {
            'img_path': img_path,
            'hflipped': hflipped
        }

        return metadatas, bev_map, targets


    def get_image(self, idx):
        img_path = os.path.join(self.image_dir, '{:06d}.png'.format(idx))
        # print(img_path)
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        return img_path, img

    def get_calib(self, idx):
        calib_file = os.path.join(self.calib_dir, '{:06d}.txt'.format(idx))
        # assert os.path.isfile(calib_file)
        return data_utils.Calibration(calib_file)

    # def get_lidar(self, idx):
    #     lidar_file = os.path.join(self.lidar_dir, '{:06d}.bin'.format(idx))
    #     # assert os.path.isfile(lidar_file)
    #     return np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 4)
    # def get_lidar(self, sample_id):
    #     lidar_file = os.path.join(self.lidar_dir, str(sample_id).zfill(6) + '.bin')
    #
    #     with open(lidar_file, 'rb') as f:
    #         print("f:",f)
    #         lidar_data = np.fromfile(f, dtype=np.float32)
    #         lidar_data = lidar_data.reshape((-1, 4))
    #     return lidar_data
    # def get_lidar(self, sample_id):
    #     lidar_file = os.path.join(self.lidar_dir, str(sample_id).zfill(6) + '.bin')
    #
    #     with open(lidar_file, 'rb') as f:
    #         print("f:",f)
    #         lidar_data = np.fromfile(f, dtype=np.float32)
    #         lidar_data = lidar_data.reshape((-1, 4))
    #     return lidar_data

    def get_lidar(self, sample_id):
        lidar_file = os.path.join(self.lidar_dir, str(sample_id).zfill(6) + '.bin')

        with open(lidar_file, 'rb') as f:
            print(f)
            lidar_data = np.fromfile(f, dtype=np.float32)
            lidar_data = lidar_data.reshape((-1, 4))

        # È¸Àü °¢µµ (½Ã°è ¹æÇâÀ¸·Î 90µµ È¸Àü)
        angle = np.radians(90)

        # È¸Àü Ã³¸® Å×½ºÆ®
        axis = np.array([0, 0, 1])  # Z\ucd95

        # È¸Àü º¯È¯ ¸ðµâ Á¦ÀÛ
        rotation_matrix = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1]
        ])

        # È¸Àü »ç¿ë
        rotated_lidar_data = np.dot(lidar_data[:, :3], rotation_matrix.T)
        rotated_lidar_data = np.hstack((rotated_lidar_data, lidar_data[:, 3:]))

        return rotated_lidar_data

    def get_label(self, idx):
        labels = []
        label_path = os.path.join(self.label_dir, '{:06d}.txt'.format(idx))

        for line in open(label_path, 'r'):
            line = line.rstrip()
            line_parts = line.split(' ')
            obj_name = line_parts[0]  # 'Car', 'Pedestrian', ...
            cat_id = int(cnf.CLASS_NAME_TO_ID[obj_name])

            if cat_id <= -99:  # ignore Tram and Misc
                continue

            try:
                truncated = int(float(line_parts[1]))  # truncated pixel ratio [0..1]
                occluded = int(line_parts[2])  # 0=visible, 1=partly occluded, 2=fully occluded, 3=unknown
                alpha = float(line_parts[3])  # object observation angle [-pi..pi]
                # xmin, ymin, xmax, ymax
                bbox = np.array(
                    [float(line_parts[4]), float(line_parts[5]), float(line_parts[6]), float(line_parts[7])])
                # height, width, length (h, w, l)
                h, w, l = float(line_parts[8]), float(line_parts[9]), float(line_parts[10])
                # location (x,y,z) in camera coord.
                x, y, z = float(line_parts[11]), float(line_parts[12]), float(line_parts[13])
                ry = float(line_parts[14])  # yaw angle (around Y-axis in camera coordinates) [-pi..pi]
                if y > 50:
                    bbox = round(bbox / 100, 2)
                    h = round(h / 100, 2)
                    w = round(w / 100, 2)
                    l = round(l / 100, 2)
                    x = round(x / 100, 2)
                    y = round(y / 100, 2)
                    z = round(z / 100, 2)
                object_label = [cat_id, x, y, z, h, w, l, -ry]
                # object_label = [cat_id, x, y, z, h, w, l, -ry]
                labels.append(object_label)
                print(object_label)
            except ValueError:
                # print("Error: Could not convert string to float or int - line: ", line)
                pass

        if len(labels) == 0:
            labels = np.zeros((1, 8), dtype=np.float32)
            has_labels = False
        else:
            labels = np.array(labels, dtype=np.float32)
            has_labels = True

        return labels, has_labels


    def build_targets(self, labels, hflipped):
        minX = cnf.boundary['minX']
        maxX = cnf.boundary['maxX']
        minY = cnf.boundary['minY']
        maxY = cnf.boundary['maxY']
        minZ = cnf.boundary['minZ']
        maxZ = cnf.boundary['maxZ']

        num_objects = min(len(labels), self.max_objects)
        hm_l, hm_w = self.hm_size

        hm_main_center = np.zeros((self.num_classes, hm_l, hm_w), dtype=np.float32)
        # print("hm_main_center_182: ", hm_main_center)
        cen_offset = np.zeros((self.max_objects, 2), dtype=np.float32)
        direction = np.zeros((self.max_objects, 2), dtype=np.float32)
        z_coor = np.zeros((self.max_objects, 1), dtype=np.float32)
        dimension = np.zeros((self.max_objects, 3), dtype=np.float32)

        indices_center = np.zeros((self.max_objects), dtype=np.int64)
        obj_mask = np.zeros((self.max_objects), dtype=np.uint8)

        for k in range(num_objects):
            cls_id, x, y, z, h, w, l, yaw = labels[k]
            cls_id = int(cls_id)
            # Invert yaw angle
            yaw = -yaw
            if not ((minX <= x <= maxX) and (minY <= y <= maxY) and (minZ <= z <= maxZ)):
                continue
            if (h <= 0) or (w <= 0) or (l <= 0):
                continue

            bbox_l = l / cnf.bound_size_x * hm_l
            bbox_w = w / cnf.bound_size_y * hm_w
            radius = data_utils.compute_radius((math.ceil(bbox_l), math.ceil(bbox_w)))
            radius = max(0, int(radius))

            center_y = (x - minX) / cnf.bound_size_x * hm_l  # x --> y (invert to 2D image space)
            center_x = (y - minY) / cnf.bound_size_y * hm_w  # y --> x
            center = np.array([center_x, center_y], dtype=np.float32)

            if hflipped:
                center[0] = hm_w - center[0] - 1

            center_int = center.astype(np.int32)
            if cls_id < 0:
                ignore_ids = [_ for _ in range(self.num_classes)] if cls_id == - 1 else [- cls_id - 2]
                # Consider to make mask ignore
                for cls_ig in ignore_ids:
                    data_utils.gen_hm_radius(hm_main_center[cls_ig], center_int, radius)
                hm_main_center[ignore_ids, center_int[1], center_int[0]] = 0.9999
                continue

            # Generate heatmaps for main center
            data_utils.gen_hm_radius(hm_main_center[cls_id], center, radius)
            # print("get_hm_radius_228 : ", gen_hm_radius(hm_main_center[cls_id], center, radius))
            # print("len_get_hm_radius_229 : ", len(gen_hm_radius(hm_main_center[cls_id], center, radius)))
            # Index of the center
            indices_center[k] = center_int[1] * hm_w + center_int[0]

            # targets for center offset
            cen_offset[k] = center - center_int

            # targets for dimension
            dimension[k, 0] = h
            dimension[k, 1] = w
            dimension[k, 2] = l

            # targets for direction
            direction[k, 0] = math.sin(float(yaw))  # im
            direction[k, 1] = math.cos(float(yaw))  # re
            # im -->> -im
            if hflipped:
                direction[k, 0] = - direction[k, 0]

            # targets for depth
            z_coor[k] = z - minZ

            # Generate object masks
            obj_mask[k] = 1

        targets = {
            'hm_cen': hm_main_center,
            'cen_offset': cen_offset,
            'direction': direction,
            'z_coor': z_coor,
            'dim': dimension,
            'indices_center': indices_center,
            'obj_mask': obj_mask,
        }

        return targets

    def draw_img_with_label(self, index):
        sample_id = int(self.sample_id_list[index])
        img_path, img_rgb = self.get_image(sample_id)
        lidarData = self.get_lidar(sample_id)
        calib = self.get_calib(sample_id)
        labels, has_labels = self.get_label(sample_id)
        print("labels: ", labels)
        if has_labels:
            labels[:, 1:] = transformation.camera_to_lidar_box(labels[:, 1:], calib.V2C, calib.R0, calib.P2)

        if self.lidar_aug:
            lidarData, labels[:, 1:] = self.lidar_aug(lidarData, labels[:, 1:])
        # print("lidarData_271 : ", lidarData)
        lidarData, labels = data_utils.get_filtered_lidar(lidarData, cnf.boundary, labels)
        # print("lidarData_273 : ", lidarData)
        lidarData = lidarData
        labels = labels
        bev_map = bev_utils.makeBEVMap(lidarData, cnf.boundary)

        return bev_map, labels, img_rgb, img_path

class data_utils:
    class Object3d(object):
        ''' 3d object label '''

        def __init__(self, label_file_line):
            data = label_file_line.split(' ')
            data[1:] = [float(x) for x in data[1:]]
            # extract label, truncation, occlusion
            self.type = data[0]  # 'Car', 'Pedestrian', ...
            self.cls_id = self.cls_type_to_id(self.type)
            self.truncation = data[1]  # truncated pixel ratio [0..1]
            self.occlusion = int(data[2])  # 0=visible, 1=partly occluded, 2=fully occluded, 3=unknown
            self.alpha = data[3]  # object observation angle [-pi..pi]

            # extract 2d bounding box in 0-based coordinates
            self.xmin = data[4]  # left
            self.ymin = data[7]  # top
            self.xmax = data[6]  # right
            self.ymax = data[5]  # bottom
            self.box2d = np.array([self.xmin, self.ymin, self.xmax, self.ymax])

            # extract 3d bounding box information
            self.h = data[8]  # box height
            self.w = data[9]  # box width
            self.l = data[10]  # box length (in meters)
            self.t = (data[11], data[12], data[13])  # location (x,y,z) in camera coord.
            self.dis_to_cam = np.linalg.norm(self.t)
            self.ry = data[14]  # yaw angle (around Y-axis in camera coordinates) [-pi..pi]
            self.score = data[15] if data.__len__() == 16 else -1.0
            self.level_str = None
            self.level = self.get_obj_level()

        def cls_type_to_id(self, cls_type):
            if cls_type not in cnf.CLASS_NAME_TO_ID.keys():
                return -1

            return cnf.CLASS_NAME_TO_ID[cls_type]

        def get_obj_level(self):
            height = float(self.box2d[3]) - float(self.box2d[1]) + 1

            if height >= 40 and self.truncation <= 0.15 and self.occlusion <= 0:
                self.level_str = 'Easy'
                return 1  # Easy
            elif height >= 25 and self.truncation <= 0.3 and self.occlusion <= 1:
                self.level_str = 'Moderate'
                return 2  # Moderate
            elif height >= 25 and self.truncation <= 0.5 and self.occlusion <= 2:
                self.level_str = 'Hard'
                return 3  # Hard
            else:
                self.level_str = 'UnKnown'
                return 4

        def print_object(self):
            print('Type, truncation, occlusion, alpha: %s, %d, %d, %f' % \
                  (self.type, self.truncation, self.occlusion, self.alpha))
            print('2d bbox (x0,y0,x1,y1): %f, %f, %f, %f' % \
                  (self.xmin, self.ymin, self.xmax, self.ymax))
            print('3d bbox h,w,l: %f, %f, %f' % \
                  (self.h, self.w, self.l))
            print('3d bbox location, ry: (%f, %f, %f), %f' % \
                  (self.t[0], self.t[1], self.t[2], self.ry))

        def to_kitti_format(self):
            kitti_str = '%s %.2f %d %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f' \
                        % (self.type, self.truncation, int(self.occlusion), self.alpha, self.box2d[0], self.box2d[1],
                           self.box2d[2], self.box2d[3], self.h, self.w, self.l, self.t[0], self.t[1], self.t[2],
                           self.ry, self.score)
            return kitti_str

    def read_label(label_filename):
        lines = [line.rstrip() for line in open(label_filename)]
        objects = [data_utils.Object3d(line) for line in lines]
        return objects

    class Calibration(object):
        ''' Calibration matrices and utils
            3d XYZ in <label>.txt are in rect camera coord.
            2d box xy are in image2 coord
            Points in <lidar>.bin are in Velodyne coord.

            y_image2 = P^2_rect * x_rect
            y_image2 = P^2_rect * R0_rect * Tr_velo_to_cam * x_velo
            x_ref = Tr_velo_to_cam * x_velo
            x_rect = R0_rect * x_ref

            P^2_rect = [f^2_u,  0,      c^2_u,  -f^2_u b^2_x;
                        0,      f^2_v,  c^2_v,  -f^2_v b^2_y;
                        0,      0,      1,      0]
                     = K * [1|t]

            image2 coord:
             ----> x-axis (u)
            |
            |
            v y-axis (v)

            velodyne coord:
            front x, left y, up z

            rect/ref camera coord:
            right x, down y, front z

            Ref (KITTI paper): http://www.cvlibs.net/publications/Geiger2013IJRR.pdf

            TODO(rqi): do matrix multiplication only once for each projection.
        '''

        def __init__(self, calib_filepath):
            calibs = self.read_calib_file(calib_filepath)
            # Projection matrix from rect camera coord to image2 coord
            self.P2 = calibs['P2']
            self.P2 = np.reshape(self.P2, [3, 4])
            self.P3 = calibs['P3']
            self.P3 = np.reshape(self.P3, [3, 4])
            # Rigid transform from Velodyne coord to reference camera coord
            self.V2C = calibs['Tr_velo2cam']
            self.V2C = np.reshape(self.V2C, [3, 4])
            # Rotation from reference camera coord to rect camera coord
            self.R0 = calibs['R_rect']
            self.R0 = np.reshape(self.R0, [3, 3])

            # Camera intrinsics and extrinsics
            self.c_u = self.P2[0, 2]
            self.c_v = self.P2[1, 2]
            self.f_u = self.P2[0, 0]
            self.f_v = self.P2[1, 1]
            self.b_x = self.P2[0, 3] / (-self.f_u)  # relative
            self.b_y = self.P2[1, 3] / (-self.f_v)

        def read_calib_file(self, filepath):
            with open(filepath) as f:
                lines = f.readlines()

            obj = lines[2].strip().split(' ')[1:]
            P2 = np.array(obj, dtype=np.float32)
            obj = lines[3].strip().split(' ')[1:]
            P3 = np.array(obj, dtype=np.float32)
            obj = lines[4].strip().split(' ')[1:]
            R0 = np.array(obj, dtype=np.float32)
            obj = lines[5].strip().split(' ')[1:]
            Tr_velo_to_cam = np.array(obj, dtype=np.float32)

            return {'P2': P2.reshape(3, 4),
                    'P3': P3.reshape(3, 4),
                    'R_rect': R0.reshape(3, 3),
                    'Tr_velo2cam': Tr_velo_to_cam.reshape(3, 4)}

        def cart2hom(self, pts_3d):
            """
            :param pts: (N, 3 or 2)
            :return pts_hom: (N, 4 or 3)
            """
            pts_hom = np.hstack((pts_3d, np.ones((pts_3d.shape[0], 1), dtype=np.float32)))
            return pts_hom

    def compute_radius(det_size, min_overlap=0.7):
        height, width = det_size

        a1 = 1
        b1 = (height + width)
        c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
        sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
        r1 = (b1 + sq1) / 2

        a2 = 4
        b2 = 2 * (height + width)
        c2 = (1 - min_overlap) * width * height
        sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
        r2 = (b2 + sq2) / 2

        a3 = 4 * min_overlap
        b3 = -2 * min_overlap * (height + width)
        c3 = (min_overlap - 1) * width * height
        sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
        r3 = (b3 + sq3) / 2

        return min(r1, r2, r3)

    def gaussian2D(shape, sigma=1):
        m, n = [(ss - 1.) / 2. for ss in shape]
        y, x = np.ogrid[-m:m + 1, -n:n + 1]
        h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
        h[h < np.finfo(h.dtype).eps * h.max()] = 0

        return h

    def gen_hm_radius(heatmap, center, radius, k=1):
        diameter = 2 * radius + 1
        gaussian = data_utils.gaussian2D((diameter, diameter), sigma=diameter / 6)

        x, y = int(center[0]), int(center[1])

        height, width = heatmap.shape[0:2]

        left, right = min(x, radius), min(width - x, radius + 1)
        top, bottom = min(y, radius), min(height - y, radius + 1)

        masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
        masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
        if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
            np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)

        return heatmap

    def get_filtered_lidar(lidar, boundary, labels=None):
        minX = boundary['minX']
        maxX = boundary['maxX']
        minY = boundary['minY']
        maxY = boundary['maxY']
        minZ = boundary['minZ']
        maxZ = boundary['maxZ']

        # Remove the point out of range x,y,z
        mask = np.where((lidar[:, 0] >= minX) & (lidar[:, 0] <= maxX) &
                        (lidar[:, 1] >= minY) & (lidar[:, 1] <= maxY) &
                        (lidar[:, 2] >= minZ) & (lidar[:, 2] <= maxZ))
        lidar = lidar[mask]
        lidar[:, 2] = lidar[:, 2] - minZ

        if labels is not None:
            label_x = (labels[:, 1] >= minX) & (labels[:, 1] < maxX)
            label_y = (labels[:, 2] >= minY) & (labels[:, 2] < maxY)
            label_z = (labels[:, 3] >= minZ) & (labels[:, 3] < maxZ)
            mask_label = label_x & label_y & label_z
            labels = labels[mask_label]
            return lidar, labels
        else:
            return lidar

    def box3d_corners_to_center(box3d_corner):
        # (N, 8, 3) -> (N, 7)
        assert box3d_corner.ndim == 3

        xyz = np.mean(box3d_corner, axis=1)

        h = abs(np.mean(box3d_corner[:, 4:, 2] - box3d_corner[:, :4, 2], axis=1, keepdims=True))
        w = (np.sqrt(np.sum((box3d_corner[:, 0, [0, 1]] - box3d_corner[:, 1, [0, 1]]) ** 2, axis=1, keepdims=True)) +
             np.sqrt(np.sum((box3d_corner[:, 2, [0, 1]] - box3d_corner[:, 3, [0, 1]]) ** 2, axis=1, keepdims=True)) +
             np.sqrt(np.sum((box3d_corner[:, 4, [0, 1]] - box3d_corner[:, 5, [0, 1]]) ** 2, axis=1, keepdims=True)) +
             np.sqrt(np.sum((box3d_corner[:, 6, [0, 1]] - box3d_corner[:, 7, [0, 1]]) ** 2, axis=1, keepdims=True))) / 4

        l = (np.sqrt(np.sum((box3d_corner[:, 0, [0, 1]] - box3d_corner[:, 3, [0, 1]]) ** 2, axis=1, keepdims=True)) +
             np.sqrt(np.sum((box3d_corner[:, 1, [0, 1]] - box3d_corner[:, 2, [0, 1]]) ** 2, axis=1, keepdims=True)) +
             np.sqrt(np.sum((box3d_corner[:, 4, [0, 1]] - box3d_corner[:, 7, [0, 1]]) ** 2, axis=1, keepdims=True)) +
             np.sqrt(np.sum((box3d_corner[:, 5, [0, 1]] - box3d_corner[:, 6, [0, 1]]) ** 2, axis=1, keepdims=True))) / 4

        yaw = (np.arctan2(box3d_corner[:, 2, 1] - box3d_corner[:, 1, 1],
                          box3d_corner[:, 2, 0] - box3d_corner[:, 1, 0]) +
               np.arctan2(box3d_corner[:, 3, 1] - box3d_corner[:, 0, 1],
                          box3d_corner[:, 3, 0] - box3d_corner[:, 0, 0]) +
               np.arctan2(box3d_corner[:, 2, 0] - box3d_corner[:, 3, 0],
                          box3d_corner[:, 3, 1] - box3d_corner[:, 2, 1]) +
               np.arctan2(box3d_corner[:, 1, 0] - box3d_corner[:, 0, 0],
                          box3d_corner[:, 0, 1] - box3d_corner[:, 1, 1]))[:, np.newaxis] / 4

        return np.concatenate([h, w, l, xyz, yaw], axis=1).reshape(-1, 7)

    def box3d_center_to_conners(box3d_center):
        h, w, l, x, y, z, yaw = box3d_center
        Box = np.array([[-l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2],
                        [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2],
                        [0, 0, 0, 0, h, h, h, h]])

        rotMat = np.array([
            [np.cos(yaw), -np.sin(yaw), 0.0],
            [np.sin(yaw), np.cos(yaw), 0.0],
            [0.0, 0.0, 1.0]])

        velo_box = np.dot(rotMat, Box)
        cornerPosInVelo = velo_box + np.tile(np.array([x, y, z]), (8, 1)).T
        box3d_corner = cornerPosInVelo.transpose()

        return box3d_corner.astype(np.float32)

class transformation:

    def angle_in_limit(angle):
        # To limit the angle in -pi/2 - pi/2
        limit_degree = 5
        while angle >= np.pi / 2:
            angle -= np.pi
        while angle < -np.pi / 2:
            angle += np.pi
        if abs(angle + np.pi / 2) < limit_degree / 180 * np.pi:
            angle = np.pi / 2
        return angle

    def camera_to_lidar(x, y, z, V2C=None, R0=None, P2=None):
        p = np.array([x, y, z, 1])
        if V2C is None or R0 is None:
            p = np.matmul(cnf.R0_inv, p)
            p = np.matmul(cnf.Tr_velo_to_cam_inv, p)
        else:
            R0_i = np.zeros((4, 4))
            R0_i[:3, :3] = R0
            R0_i[3, 3] = 1
            p = np.matmul(np.linalg.inv(R0_i), p)
            p = np.matmul(transformation.inverse_rigid_trans(V2C), p)
        p = p[0:3]
        return tuple(p)

    def lidar_to_camera(x, y, z, V2C=None, R0=None, P2=None):
        p = np.array([x, y, z, 1])
        if V2C is None or R0 is None:
            p = np.matmul(cnf.Tr_velo_to_cam, p)
            p = np.matmul(cnf.R0, p)
        else:
            p = np.matmul(V2C, p)
            p = np.matmul(R0, p)
        p = p[0:3]
        return tuple(p)

    def camera_to_lidar_point(points):
        # (N, 3) -> (N, 3)
        N = points.shape[0]
        points = np.hstack([points, np.ones((N, 1))]).T  # (N,4) -> (4,N)

        points = np.matmul(cnf.R0_inv, points)
        points = np.matmul(cnf.Tr_velo_to_cam_inv, points).T  # (4, N) -> (N, 4)
        points = points[:, 0:3]
        return points.reshape(-1, 3)

    def lidar_to_camera_point(points, V2C=None, R0=None):
        # (N, 3) -> (N, 3)
        N = points.shape[0]
        points = np.hstack([points, np.ones((N, 1))]).T

        if V2C is None or R0 is None:
            points = np.matmul(cnf.Tr_velo_to_cam, points)
            points = np.matmul(cnf.R0, points).T
        else:
            points = np.matmul(V2C, points)
            points = np.matmul(R0, points).T
        points = points[:, 0:3]
        return points.reshape(-1, 3)

    def camera_to_lidar_box(boxes, V2C=None, R0=None, P2=None):
        # (N, 7) -> (N, 7) x,y,z,h,w,l,r
        ret = []
        for box in boxes:
            x, y, z, h, w, l, ry = box
            (x, y, z), h, w, l, rz = transformation.camera_to_lidar(x, y, z, V2C=V2C, R0=R0, P2=P2), h, w, l, -ry - np.pi / 2
            # rz = angle_in_limit(rz)
            ret.append([x, y, z, h, w, l, rz])
        return np.array(ret).reshape(-1, 7)

    def lidar_to_camera_box(boxes, V2C=None, R0=None, P2=None):
        # (N, 7) -> (N, 7) x,y,z,h,w,l,r
        ret = []
        for box in boxes:
            x, y, z, h, w, l, rz = box
            (x, y, z), h, w, l, ry = transformation.lidar_to_camera(x, y, z, V2C=V2C, R0=R0, P2=P2), h, w, l, -rz - np.pi / 2
            # ry = angle_in_limit(ry)
            ret.append([x, y, z, h, w, l, ry])
        return np.array(ret).reshape(-1, 7)

    def center_to_corner_box2d(boxes_center, coordinate='lidar'):
        # (N, 5) -> (N, 4, 2)
        N = boxes_center.shape[0]
        boxes3d_center = np.zeros((N, 7))
        boxes3d_center[:, [0, 1, 4, 5, 6]] = boxes_center
        boxes3d_corner = transformation.center_to_corner_box3d(boxes3d_center, coordinate=coordinate)

        return boxes3d_corner[:, 0:4, 0:2]

    def center_to_corner_box3d(boxes_center, coordinate='lidar'):
        # (N, 7) -> (N, 8, 3)
        N = boxes_center.shape[0]
        ret = np.zeros((N, 8, 3), dtype=np.float32)

        if coordinate == 'camera':
            boxes_center = transformation.camera_to_lidar_box(boxes_center)

        for i in range(N):
            box = boxes_center[i]
            translation = box[0:3]
            size = box[3:6]
            rotation = [0, 0, box[-1]]

            h, w, l = size[0], size[1], size[2]
            trackletBox = np.array([  # in velodyne coordinates around zero point and without orientation yet
                [-l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2], \
                [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2], \
                [0, 0, 0, 0, h, h, h, h]])

            # re-create 3D bounding box in velodyne coordinate system
            yaw = rotation[2]
            rotMat = np.array([
                [np.cos(yaw), -np.sin(yaw), 0.0],
                [np.sin(yaw), np.cos(yaw), 0.0],
                [0.0, 0.0, 1.0]])
            cornerPosInVelo = np.dot(rotMat, trackletBox) + np.tile(translation, (8, 1)).T
            box3d = cornerPosInVelo.transpose()
            ret[i] = box3d

        if coordinate == 'camera':
            for idx in range(len(ret)):
                ret[idx] = transformation.lidar_to_camera_point(ret[idx])

        return ret

    CORNER2CENTER_AVG = True

    def corner_to_center_box3d(boxes_corner, coordinate='camera'):
        # (N, 8, 3) -> (N, 7) x,y,z,h,w,l,ry/z
        if coordinate == 'lidar':
            for idx in range(len(boxes_corner)):
                boxes_corner[idx] = transformation.lidar_to_camera_point(boxes_corner[idx])

        ret = []
        for roi in boxes_corner:
            if transformation.CORNER2CENTER_AVG:  # average version
                roi = np.array(roi)
                h = abs(np.sum(roi[:4, 1] - roi[4:, 1]) / 4)
                w = np.sum(
                    np.sqrt(np.sum((roi[0, [0, 2]] - roi[3, [0, 2]]) ** 2)) +
                    np.sqrt(np.sum((roi[1, [0, 2]] - roi[2, [0, 2]]) ** 2)) +
                    np.sqrt(np.sum((roi[4, [0, 2]] - roi[7, [0, 2]]) ** 2)) +
                    np.sqrt(np.sum((roi[5, [0, 2]] - roi[6, [0, 2]]) ** 2))
                ) / 4
                l = np.sum(
                    np.sqrt(np.sum((roi[0, [0, 2]] - roi[1, [0, 2]]) ** 2)) +
                    np.sqrt(np.sum((roi[2, [0, 2]] - roi[3, [0, 2]]) ** 2)) +
                    np.sqrt(np.sum((roi[4, [0, 2]] - roi[5, [0, 2]]) ** 2)) +
                    np.sqrt(np.sum((roi[6, [0, 2]] - roi[7, [0, 2]]) ** 2))
                ) / 4
                x = np.sum(roi[:, 0], axis=0) / 8
                y = np.sum(roi[0:4, 1], axis=0) / 4
                z = np.sum(roi[:, 2], axis=0) / 8
                ry = np.sum(
                    math.atan2(roi[2, 0] - roi[1, 0], roi[2, 2] - roi[1, 2]) +
                    math.atan2(roi[6, 0] - roi[5, 0], roi[6, 2] - roi[5, 2]) +
                    math.atan2(roi[3, 0] - roi[0, 0], roi[3, 2] - roi[0, 2]) +
                    math.atan2(roi[7, 0] - roi[4, 0], roi[7, 2] - roi[4, 2]) +
                    math.atan2(roi[0, 2] - roi[1, 2], roi[1, 0] - roi[0, 0]) +
                    math.atan2(roi[4, 2] - roi[5, 2], roi[5, 0] - roi[4, 0]) +
                    math.atan2(roi[3, 2] - roi[2, 2], roi[2, 0] - roi[3, 0]) +
                    math.atan2(roi[7, 2] - roi[6, 2], roi[6, 0] - roi[7, 0])
                ) / 8
                if w > l:
                    w, l = l, w
                    ry = ry - np.pi / 2
                elif l > w:
                    l, w = w, l
                    ry = ry - np.pi / 2
                ret.append([x, y, z, h, w, l, ry])

            else:  # max version
                h = max(abs(roi[:4, 1] - roi[4:, 1]))
                w = np.max(
                    np.sqrt(np.sum((roi[0, [0, 2]] - roi[3, [0, 2]]) ** 2)) +
                    np.sqrt(np.sum((roi[1, [0, 2]] - roi[2, [0, 2]]) ** 2)) +
                    np.sqrt(np.sum((roi[4, [0, 2]] - roi[7, [0, 2]]) ** 2)) +
                    np.sqrt(np.sum((roi[5, [0, 2]] - roi[6, [0, 2]]) ** 2))
                )
                l = np.max(
                    np.sqrt(np.sum((roi[0, [0, 2]] - roi[1, [0, 2]]) ** 2)) +
                    np.sqrt(np.sum((roi[2, [0, 2]] - roi[3, [0, 2]]) ** 2)) +
                    np.sqrt(np.sum((roi[4, [0, 2]] - roi[5, [0, 2]]) ** 2)) +
                    np.sqrt(np.sum((roi[6, [0, 2]] - roi[7, [0, 2]]) ** 2))
                )
                x = np.sum(roi[:, 0], axis=0) / 8
                y = np.sum(roi[0:4, 1], axis=0) / 4
                z = np.sum(roi[:, 2], axis=0) / 8
                ry = np.sum(
                    math.atan2(roi[2, 0] - roi[1, 0], roi[2, 2] - roi[1, 2]) +
                    math.atan2(roi[6, 0] - roi[5, 0], roi[6, 2] - roi[5, 2]) +
                    math.atan2(roi[3, 0] - roi[0, 0], roi[3, 2] - roi[0, 2]) +
                    math.atan2(roi[7, 0] - roi[4, 0], roi[7, 2] - roi[4, 2]) +
                    math.atan2(roi[0, 2] - roi[1, 2], roi[1, 0] - roi[0, 0]) +
                    math.atan2(roi[4, 2] - roi[5, 2], roi[5, 0] - roi[4, 0]) +
                    math.atan2(roi[3, 2] - roi[2, 2], roi[2, 0] - roi[3, 0]) +
                    math.atan2(roi[7, 2] - roi[6, 2], roi[6, 0] - roi[7, 0])
                ) / 8
                if w > l:
                    w, l = l, w
                    ry = transformation.angle_in_limit(ry + np.pi / 2)
                ret.append([x, y, z, h, w, l, ry])

        if coordinate == 'lidar':
            ret = transformation.camera_to_lidar_box(np.array(ret))

        return np.array(ret)

    def point_transform(points, tx, ty, tz, rx=0, ry=0, rz=0):
        # Input:
        #   points: (N, 3)
        #   rx/y/z: in radians
        # Output:
        #   points: (N, 3)
        N = points.shape[0]
        points = np.hstack([points, np.ones((N, 1))])

        mat1 = np.eye(4)
        mat1[3, 0:3] = tx, ty, tz
        points = np.matmul(points, mat1)

        if rx != 0:
            mat = np.zeros((4, 4))
            mat[0, 0] = 1
            mat[3, 3] = 1
            mat[1, 1] = np.cos(rx)
            mat[1, 2] = -np.sin(rx)
            mat[2, 1] = np.sin(rx)
            mat[2, 2] = np.cos(rx)
            points = np.matmul(points, mat)

        if ry != 0:
            mat = np.zeros((4, 4))
            mat[1, 1] = 1
            mat[3, 3] = 1
            mat[0, 0] = np.cos(ry)
            mat[0, 2] = np.sin(ry)
            mat[2, 0] = -np.sin(ry)
            mat[2, 2] = np.cos(ry)
            points = np.matmul(points, mat)

        if rz != 0:
            mat = np.zeros((4, 4))
            mat[2, 2] = 1
            mat[3, 3] = 1
            mat[0, 0] = np.cos(rz)
            mat[0, 1] = -np.sin(rz)
            mat[1, 0] = np.sin(rz)
            mat[1, 1] = np.cos(rz)
            points = np.matmul(points, mat)

        return points[:, 0:3]

    def box_transform(boxes, tx, ty, tz, r=0, coordinate='lidar'):
        # Input:
        #   boxes: (N, 7) x y z h w l rz/y
        # Output:
        #   boxes: (N, 7) x y z h w l rz/y
        boxes_corner = transformation.center_to_corner_box3d(boxes, coordinate=coordinate)  # (N, 8, 3)
        for idx in range(len(boxes_corner)):
            if coordinate == 'lidar':
                boxes_corner[idx] = transformation.point_transform(boxes_corner[idx], tx, ty, tz, rz=r)
            else:
                boxes_corner[idx] = transformation.point_transform(boxes_corner[idx], tx, ty, tz, ry=r)

        return transformation.corner_to_center_box3d(boxes_corner, coordinate=coordinate)

    def inverse_rigid_trans(Tr):
        ''' Inverse a rigid body transform matrix (3x4 as [R|t])
            [R'|-R't; 0|1]
        '''
        inv_Tr = np.zeros_like(Tr)  # 3x4
        inv_Tr[0:3, 0:3] = np.transpose(Tr[0:3, 0:3])
        inv_Tr[0:3, 3] = np.dot(-np.transpose(Tr[0:3, 0:3]), Tr[0:3, 3])
        return inv_Tr

    class Compose(object):
        def __init__(self, transforms, p=1.0):
            self.transforms = transforms
            self.p = p

        def __call__(self, lidar, labels):
            if np.random.random() <= self.p:
                for t in self.transforms:
                    lidar, labels = t(lidar, labels)
            return lidar, labels

    class OneOf(object):
        def __init__(self, transforms, p=1.0):
            self.transforms = transforms
            self.p = p

        def __call__(self, lidar, labels):
            if np.random.random() <= self.p:
                choice = np.random.randint(low=0, high=len(self.transforms))
                lidar, labels = self.transforms[choice](lidar, labels)

            return lidar, labels

    class Random_Rotation(object):
        def __init__(self, limit_angle=np.pi / 4, p=0.5):
            self.limit_angle = limit_angle
            self.p = p

        def __call__(self, lidar, labels):
            """
            :param labels: # (N', 7) x, y, z, h, w, l, r
            :return:
            """
            if np.random.random() <= self.p:
                angle = np.random.uniform(-self.limit_angle, self.limit_angle)
                lidar[:, 0:3] = transformation.point_transform(lidar[:, 0:3], 0, 0, 0, rz=angle)
                labels = transformation.box_transform(labels, 0, 0, 0, r=angle, coordinate='lidar')

            return lidar, labels

    class Random_Scaling(object):
        def __init__(self, scaling_range=(0.95, 1.05), p=0.5):
            self.scaling_range = scaling_range
            self.p = p

        def __call__(self, lidar, labels):
            """
            :param labels: # (N', 7) x, y, z, h, w, l, r
            :return:
            """
            if np.random.random() <= self.p:
                factor = np.random.uniform(self.scaling_range[0], self.scaling_range[0])
                lidar[:, 0:3] = lidar[:, 0:3] * factor
                labels[:, 0:6] = labels[:, 0:6] * factor

            return lidar, labels

    class Cutout(object):
        """Randomly mask out one or more patches from an image.
        Args:
            n_holes (int): Number of patches to cut out of each image.
            length (int): The length (in pixels) of each square patch.
            Refer from: https://github.com/uoguelph-mlrg/Cutout/blob/master/util/cutout.py
        """

        def __init__(self, n_holes, ratio, fill_value=0., p=1.0):
            self.n_holes = n_holes
            self.ratio = ratio
            assert 0. <= fill_value <= 1., "the fill value is in a range of 0 to 1"
            self.fill_value = fill_value
            self.p = p

        def __call__(self, img, targets):
            """
            Args:
                img (Tensor): Tensor image of size (C, H, W).
            Returns:
                Tensor: Image with n_holes of dimension length x length cut out of it.
            """
            if np.random.random() <= self.p:
                h = img.size(1)
                w = img.size(2)

                h_cutout = int(self.ratio * h)
                w_cutout = int(self.ratio * w)

                for n in range(self.n_holes):
                    y = np.random.randint(h)
                    x = np.random.randint(w)

                    y1 = np.clip(y - h_cutout // 2, 0, h)
                    y2 = np.clip(y + h_cutout // 2, 0, h)
                    x1 = np.clip(x - w_cutout // 2, 0, w)
                    x2 = np.clip(x + w_cutout // 2, 0, w)

                    img[:, y1: y2, x1: x2] = self.fill_value  # Zero out the selected area
                    # Remove targets that are in the selected area
                    keep_target = []
                    for target_idx, target in enumerate(targets):
                        _, _, target_x, target_y, target_w, target_l, _, _ = target
                        if (x1 <= target_x * w <= x2) and (y1 <= target_y * h <= y2):
                            continue
                        keep_target.append(target_idx)
                    targets = targets[keep_target]

            return img, targets

class evaluation_utils:

    def _nms(heat, kernel=3):
        pad = (kernel - 1) // 2
        hmax = F.max_pool2d(heat, (kernel, kernel), stride=1, padding=pad)
        keep = (hmax == heat).float()

        return heat * keep

    def _gather_feat(feat, ind, mask=None):
        dim = feat.size(2)
        ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
        feat = feat.gather(1, ind)
        if mask is not None:
            mask = mask.unsqueeze(2).expand_as(feat)
            feat = feat[mask]
            feat = feat.view(-1, dim)
        return feat

    def _transpose_and_gather_feat(feat, ind):
        feat = feat.permute(0, 2, 3, 1).contiguous()
        feat = feat.view(feat.size(0), -1, feat.size(3))
        feat = evaluation_utils._gather_feat(feat, ind)
        return feat

    def _topk(scores, K=40):
        batch, cat, height, width = scores.size()

        topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

        topk_inds = topk_inds % (height * width)
        topk_ys = (torch.floor_divide(topk_inds, width)).float()
        topk_xs = (topk_inds % width).int().float()

        topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
        topk_clses = (torch.floor_divide(topk_ind, K)).int()
        topk_inds = evaluation_utils._gather_feat(topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
        topk_ys = evaluation_utils._gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
        topk_xs = evaluation_utils._gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

        return topk_score, topk_inds, topk_clses, topk_ys, topk_xs

    def _topk_channel(scores, K=40):
        batch, cat, height, width = scores.size()

        topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

        topk_inds = topk_inds % (height * width)
        topk_ys = (topk_inds / width).int().float()
        topk_xs = (topk_inds % width).int().float()

        return topk_scores, topk_inds, topk_ys, topk_xs

    def decode(hm_cen, cen_offset, direction, z_coor, dim, K=40):
        batch_size, num_classes, height, width = hm_cen.size()

        hm_cen = evaluation_utils._nms(hm_cen)
        scores, inds, clses, ys, xs = evaluation_utils._topk(hm_cen, K=K)
        if cen_offset is not None:
            cen_offset = evaluation_utils._transpose_and_gather_feat(cen_offset, inds)
            cen_offset = cen_offset.view(batch_size, K, 2)
            xs = xs.view(batch_size, K, 1) + cen_offset[:, :, 0:1]
            ys = ys.view(batch_size, K, 1) + cen_offset[:, :, 1:2]
        else:
            xs = xs.view(batch_size, K, 1) + 0.5
            ys = ys.view(batch_size, K, 1) + 0.5

        direction = evaluation_utils._transpose_and_gather_feat(direction, inds)
        direction = direction.view(batch_size, K, 2)
        z_coor = evaluation_utils._transpose_and_gather_feat(z_coor, inds)
        z_coor = z_coor.view(batch_size, K, 1)
        dim = evaluation_utils._transpose_and_gather_feat(dim, inds)
        dim = dim.view(batch_size, K, 3)
        clses = clses.view(batch_size, K, 1).float()
        scores = scores.view(batch_size, K, 1)

        # (scores x 1, ys x 1, xs x 1, z_coor x 1, dim x 3, direction x 2, clses x 1)
        # (scores-0:1, ys-1:2, xs-2:3, z_coor-3:4, dim-4:7, direction-7:9, clses-9:10)
        # detections: [batch_size, K, 10]
        detections = torch.cat([scores, xs, ys, z_coor, dim, direction, clses], dim=2)

        return detections

    def get_yaw(direction):
        return np.arctan2(direction[:, 0:1], direction[:, 1:2])

    def post_processing(detections, num_classes=3, down_ratio=4, peak_thresh=0.2):
        """
        :param detections: [batch_size, K, 10]
        # (scores x 1, xs x 1, ys x 1, z_coor x 1, dim x 3, direction x 2, clses x 1)
        # (scores-0:1, xs-1:2, ys-2:3, z_coor-3:4, dim-4:7, direction-7:9, clses-9:10)
        :return:
        """
        # TODO: Need to consider rescale to the original scale: x, y

        ret = []
        for i in range(detections.shape[0]):
            top_preds = {}
            classes = detections[i, :, -1]
            for j in range(num_classes):
                inds = (classes == j)
                # x, y, z, h, w, l, yaw
                top_preds[j] = np.concatenate([
                    detections[i, inds, 0:1],
                    detections[i, inds, 1:2] * down_ratio,
                    detections[i, inds, 2:3] * down_ratio,
                    detections[i, inds, 3:4],
                    detections[i, inds, 4:5],
                    detections[i, inds, 5:6] / cnf.bound_size_y * cnf.BEV_WIDTH,
                    detections[i, inds, 6:7] / cnf.bound_size_x * cnf.BEV_HEIGHT,
                    evaluation_utils.get_yaw(detections[i, inds, 7:9]).astype(np.float32)], axis=1)
                # Filter by peak_thresh
                if len(top_preds[j]) > 0:
                    keep_inds = (top_preds[j][:, 0] > peak_thresh)
                    top_preds[j] = top_preds[j][keep_inds]
            ret.append(top_preds)

        return ret

    def draw_predictions(img, detections, num_classes=3):
        for j in range(num_classes):
            if len(detections[j]) > 0:
                for det in detections[j]:
                    # (scores-0:1, x-1:2, y-2:3, z-3:4, dim-4:7, yaw-7:8)
                    _score, _x, _y, _z, _h, _w, _l, _yaw = det
                    bev_utils.drawRotatedBox(img, _x, _y, _w, _l, _yaw, cnf.colors[int(j)])

        return img

    # \uc774\ubd80\ubd84 \uccb4\ud06c\ud574\uc8fc\uc138\uc694!!!
    def convert_det_to_real_values(detections, num_classes=8):
        kitti_dets = []
        for cls_id in range(num_classes):
            print("cls_id_170 : ", cls_id)
            print("detections[cls_id]_172 : ", detections[cls_id])
            if len(detections[cls_id]) > 0:

                for det in detections[cls_id]:
                    # (scores-0:1, x-1:2, y-2:3, z-3:4, dim-4:7, yaw-7:8)
                    _score, _x, _y, _z, _h, _w, _l, _yaw = det
                    _yaw = -_yaw
                    x = _y / cnf.BEV_HEIGHT * cnf.bound_size_x + cnf.boundary['minX']
                    y = _x / cnf.BEV_WIDTH * cnf.bound_size_y + cnf.boundary['minY']
                    z = _z + cnf.boundary['minZ']
                    w = _w / cnf.BEV_WIDTH * cnf.bound_size_y
                    l = _l / cnf.BEV_HEIGHT * cnf.bound_size_x

                    kitti_dets.append([cls_id, x, y, z, _h, w, l, _yaw])

        return np.array(kitti_dets)

class visualization_utils:

    def roty(angle):
        # Rotation about the y-axis.
        c = np.cos(angle)
        s = np.sin(angle)
        return np.array([[c, 0, s],
                         [0, 1, 0],
                         [-s, 0, c]])

    def compute_box_3d(dim, location, ry):
        # dim: 3
        # location: 3
        # ry: 1
        # return: 8 x 3
        R = visualization_utils.roty(ry)
        h, w, l = dim
        x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
        y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
        z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]

        corners = np.array([x_corners, y_corners, z_corners], dtype=np.float32)
        corners_3d = np.dot(R, corners)
        corners_3d = corners_3d + np.array(location, dtype=np.float32).reshape(3, 1)
        return corners_3d.transpose(1, 0)

    def project_to_image(pts_3d, P):
        # pts_3d: n x 3
        # P: 3 x 4
        # return: n x 2
        pts_3d_homo = np.concatenate([pts_3d, np.ones((pts_3d.shape[0], 1), dtype=np.float32)], axis=1)
        pts_2d = np.dot(P, pts_3d_homo.transpose(1, 0)).transpose(1, 0)
        pts_2d = pts_2d[:, :2] / pts_2d[:, 2:]

        return pts_2d.astype(np.int)

    def draw_box_3d_v2(image, qs, color=(255, 0, 255), thickness=2):
        ''' Draw 3d bounding box in image
            qs: (8,3) array of vertices for the 3d box in following order:
                1 -------- 0
               /|         /|
              2 -------- 3 .
              | |        | |
              . 5 -------- 4
              |/         |/
              6 -------- 7

        '''
        qs = qs.astype(np.int32)
        for k in range(0, 4):
            # Ref: http://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html
            i, j = k, (k + 1) % 4
            # use LINE_AA for opencv3
            cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness)
            i, j = k + 4, (k + 1) % 4 + 4
            cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness)

            i, j = k, k + 4
            cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness)

        return image

    def draw_box_3d(image, corners, color=(0, 0, 255)):
        ''' Draw 3d bounding box in image
            corners: (8,3) array of vertices for the 3d box in following order:
                1 -------- 0
               /|         /|
              2 -------- 3 .
              | |        | |
              . 5 -------- 4
              |/         |/
              6 -------- 7

        '''

        face_idx = [[0, 1, 5, 4],
                    [1, 2, 6, 5],
                    [2, 3, 7, 6],
                    [3, 0, 4, 7]]
        for ind_f in range(3, -1, -1):
            f = face_idx[ind_f]
            for j in range(4):
                cv2.line(image, (corners[f[j], 0], corners[f[j], 1]),
                         (corners[f[(j + 1) % 4], 0], corners[f[(j + 1) % 4], 1]), color, 2, lineType=cv2.LINE_AA)
            if ind_f == 0:
                cv2.line(image, (corners[f[0], 0], corners[f[0], 1]),
                         (corners[f[2], 0], corners[f[2], 1]), color, 1, lineType=cv2.LINE_AA)
                cv2.line(image, (corners[f[1], 0], corners[f[1], 1]),
                         (corners[f[3], 0], corners[f[3], 1]), color, 1, lineType=cv2.LINE_AA)

        return image

    def show_rgb_image_with_boxes(img, labels, calib):
        for box_idx, label in enumerate(labels):
            cls_id, location, dim, ry = label[0], label[1:4], label[4:7], label[7]
            # cls_id, location, dim, ry = label[0], label[1:4], label[4:7], label[7]

            if location[2] < 10. - 0:  # The object is too close to the camera, ignore it during visualization
                continue
            if cls_id < 0:
                continue
            corners_3d = visualization_utils.compute_box_3d(dim, location, ry)
            corners_2d = visualization_utils.project_to_image(corners_3d, calib.P2)
            img = visualization_utils.draw_box_3d(img, corners_2d, color=cnf.colors[int(cls_id)])
            # img = draw_box_3d(img, corners_2d, color=cnf.colors[int(cls_id)]) <- \uc774 \ubd80\ubd84\uc774 \ucd9c\ub825 \uacb0\uacfc\uc758 \uc774\ubbf8\uc9c0 \ubd80\ubd84 \uc0c9\uc0c1 \ubcc0\uacbd \ud30c\ud2b8\uc784

        return img

    def merge_rgb_to_bev(img_rgb, img_bev, output_width):
        img_rgb_h, img_rgb_w = img_rgb.shape[:2]
        ratio_rgb = output_width / img_rgb_w
        output_rgb_h = int(ratio_rgb * img_rgb_h)
        ret_img_rgb = cv2.resize(img_rgb, (output_width, output_rgb_h))

        img_bev_h, img_bev_w = img_bev.shape[:2]
        # img_bev_h, img_bev_w = img_bev.shape[:2]
        ratio_bev = output_width / img_bev_w
        output_bev_h = int(img_bev_h)

        ret_img_bev = cv2.resize(img_bev, (output_width, output_bev_h))
        out_img = np.zeros((output_rgb_h + output_bev_h, output_width, 3), dtype=np.uint8)
        # out_img = np.zeros((output_rgb_h + output_bev_h, output_width, 3), dtype=np.uint8)

        # Upper: RGB --> BEV
        out_img[:output_rgb_h, ...] = ret_img_rgb
        out_img[output_rgb_h:, ...] = ret_img_bev
        return out_img
        # return ret_img_bev

# def main():
#     setproctitle('Lidar Detection')
#     cnt = 0
#     configs = edict()
#     configs.distributed = False  # For testing
#     configs.pin_memory = False
#     configs.num_samples = None
#     configs.input_size = (608, 608)
#     configs.hm_size = (152, 152)
#     configs.max_objects = 50
#     configs.num_classes = 8
#     configs.output_width = 608
#
#     configs.dataset_dir = os.path.join('../../', 'dataset', 'kitti')
#     # lidar_aug = OneOf([
#     #     Random_Rotation(limit_angle=np.pi / 4, p=1.),
#     #     Random_Scaling(scaling_range=(0.95, 1.05), p=1.),
#     # ], p=1.)
#     lidar_aug = None
#
#     dataset = KittiDataset(configs, mode='val', lidar_aug=lidar_aug, hflip_prob=0., num_samples=configs.num_samples)
#     cnt = 0
#     for idx in range(len(dataset)):
#         bev_map, labels, img_rgb, img_path = dataset.draw_img_with_label(idx)
#         calib = Calibration(img_path.replace(".png", ".txt").replace("image_2", "calib"))
#         bev_map = (bev_map.transpose(1, 2, 0) * 255).astype(np.uint8)
#         bev_map = cv2.resize(bev_map, (cnf.BEV_HEIGHT, cnf.BEV_WIDTH))
#         # bev_map = cv2.resize(bev_map, (cnf.BEV_HEIGHT, cnf.BEV_WIDTH))
#         for box_idx, (cls_id, x, y, z, h, w, l, yaw) in enumerate(labels):
#             # Draw rotated box
#             yaw = -yaw
#             y1 = int((x - cnf.boundary['minX']) / cnf.DISCRETIZATION)
#             x1 = int((y - cnf.boundary['minY']) / cnf.DISCRETIZATION)
#             w1 = int(w / cnf.DISCRETIZATION)
#             l1 = int(l / cnf.DISCRETIZATION)
#
#             drawRotatedBox(bev_map, x1, y1, w1, l1, yaw, cnf.colors[int(cls_id)])
#         # Rotate the bev_map
#         # bev_map = cv2.rotate(bev_map, cv2.ROTATE_180)
#
#         labels[:, 1:] = lidar_to_camera_box(labels[:, 1:], calib.V2C, calib.R0, calib.P2)
#         img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
#         img_rgb = show_rgb_image_with_boxes(img_rgb, labels, calib)
#
#         out_img = merge_rgb_to_bev(img_rgb, bev_map, output_width=configs.output_width)
#
#         test_file = "../../dataset/kitti/ImageSets/train.txt"
#         output_folder = os.path.join(os.getcwd(), "output_dataset/")
#         with open(test_file, "r") as f:
#             img_names = [line.strip() for line in f.readlines()]
#             out_path = os.path.join(output_folder, img_names[cnt] + ".jpg")
#             cv2.imwrite(out_path, out_img)
#             cnt += 1
#
#         # folder_path='../../dataset/kitti/training/velodyne'
#         # # print(folder_path)
#         # for filename in os.listdir(folder_path):
#         #     if os.path.isfile(os.path.join(folder_path, filename)):
#         #         path = os.path.join(os.getcwd(), "output_dataset")zxscdzZhjhgf
#         #         path = os.path.join(path, filename + ".jpg")
#         #         cv2.imwrite(path, out_img)
#
#         if cv2.waitKey(0) & 0xff == 27:
#             break
