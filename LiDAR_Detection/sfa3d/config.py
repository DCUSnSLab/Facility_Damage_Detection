import math
import numpy as np
import os
import argparse
import torch

class test_config(object):
    CLASS_NAME_TO_ID = {
        'select': -1,
        'bollard': 0,
        'no_jaywalking_facility': 1,
        # 'jaywalking_prevention_facility': 1,
        'traffic_Light': 2,
        'road_safety_sign': 3,
        'stop_sign': 4,
        'Car': 6,
        'Van': 6,
        'Truck': 6,
    }

    colors = [[0, 255, 255], [0, 0, 255], [255, 0, 0], [255, 120, 0],
              [255, 120, 120], [0, 120, 0], [245, 10, 174], [120, 0, 255]]

    #####################################################################################

    boundary = {
        "minX": -50,
        "maxX": 50,
        "minY": -50,
        "maxY": 50,
        "minZ": -2.73,
        "maxZ": 1.27
    }

    bound_size_x = boundary['maxX'] - boundary['minX']
    bound_size_y = boundary['maxY'] - boundary['minY']
    bound_size_z = boundary['maxZ'] - boundary['minZ']

    boundary = {
        "minX": -50,
        "maxX": 50,
        "minY": -50,
        "maxY": 50,
        "minZ": -2.73,
        "maxZ": 1.27
    }
    BEV_WIDTH = 608  # across y axis -25m ~ 25m
    BEV_HEIGHT = 608  # across x axis 0m ~ 50m
    DISCRETIZATION = (boundary["maxX"] - boundary["minX"]) / BEV_HEIGHT

    # maximum number of points per voxel
    T = 35

    # voxel size
    vd = 0.1  # z
    vh = 0.05  # y
    vw = 0.05  # x

    # voxel grid
    W = math.ceil(bound_size_x / vw)
    H = math.ceil(bound_size_y / vh)
    D = math.ceil(bound_size_z / vd)

    # Following parameters are calculated as an average from KITTI dataset for simplicity
    #####################################################################################
    Tr_velo_to_cam = np.array([
        [7.49916597e-03, -9.99971248e-01, -8.65110297e-04, -6.71807577e-03],
        [1.18652889e-02, 9.54520517e-04, -9.99910318e-01, -7.33152811e-02],
        [9.99882833e-01, 7.49141178e-03, 1.18719929e-02, -2.78557062e-01],
        [0, 0, 0, 1]
    ])

    # cal mean from train set
    R0 = np.array([
        [0.99992475, 0.00975976, -0.00734152, 0],
        [-0.0097913, 0.99994262, -0.00430371, 0],
        [0.00729911, 0.0043753, 0.99996319, 0],
        [0, 0, 0, 1]
    ])

    P2 = np.array([[719.787081, 0., 608.463003, 44.9538775],
                   [0., 719.787081, 174.545111, 0.1066855],
                   [0., 0., 1., 3.0106472e-03],
                   [0., 0., 0., 0]
                   ])

    R0_inv = np.linalg.inv(R0)
    Tr_velo_to_cam_inv = np.linalg.inv(Tr_velo_to_cam)
    P2_inv = np.linalg.pinv(P2)
    #####################################################################################

class train_config:
    def parse_train_configs(self):
        configs = argparse.Namespace(
            seed=2020,
            saved_fn="fpn_resnet_18",
            root_dir="../",
            arch="fpn_resnet_18",
            pretrained_path=None,
            ##############     Dataloader and Running configs            #######
            hflip_prob=0.5,
            no_val=False,
            num_samples=None,
            num_workers=4,
            batch_size=16,
            print_freq=50,
            # tensorboard_freq = 50,
            checkpoint_freq=1,  # frequency of saving checkpoints (default: 5)
            ##############     Training strategy            ####################
            start_epoch=1,
            num_epochs=300,
            lr_type="cosin",
            lr=0.001,
            minimum_lr=1e-7,
            momentum=0.949,
            weight_decay=0.,
            optimizer_type="adam",
            setps=[150, 180],
            ##############     Distributed Data Parallel            ############
            word_size=-1,
            rank=-1,
            # dist_url = 'tcp://127.0.0.1:29500',
            dist_backend="nccl",
            gpu_idx=0,
            no_cuda=False,
            multiprocessing_distributed=False,
            ##############     Evaluation configurations     ###################
            evalute=False,
            resuem_path=None,
            K=50,
        )

        ####################################################################
        ############## Hardware configurations #############################
        ####################################################################
        configs.device = torch.device('cpu' if configs.no_cuda else 'cuda')
        configs.ngpus_per_node = torch.cuda.device_count()

        configs.pin_memory = False
        configs.input_size = (608, 608)
        configs.hm_size = (152, 152)
        configs.down_ratio = 4
        configs.max_objects = 50

        configs.imagenet_pretrained = True
        configs.head_conv = 64
        configs.num_classes = 8
        configs.num_center_offset = 2
        configs.num_z = 1
        configs.num_dim = 3
        configs.num_direction = 2  # sin, cos

        configs.heads = {
            'hm_cen': configs.num_classes,
            'cen_offset': configs.num_center_offset,
            'direction': configs.num_direction,
            'z_coor': configs.num_z,
            'dim': configs.num_dim
        }

        configs.num_input_features = 4

        ####################################################################
        ############## Dataset, logs, Checkpoints dir ######################
        ####################################################################
        configs.dataset_dir = os.path.join(configs.root_dir, 'dataset', 'kitti')
        configs.checkpoints_dir = os.path.join(configs.root_dir, 'checkpoints', configs.saved_fn)
        configs.logs_dir = os.path.join(configs.root_dir, 'logs', configs.saved_fn)

        if not os.path.isdir(configs.checkpoints_dir):
            os.makedirs(configs.checkpoints_dir)
        if not os.path.isdir(configs.logs_dir):
            os.makedirs(configs.logs_dir)
        return configs


