import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import argparse
import sys
import os
import time
from easydict import EasyDict as edict
import cv2
import torch
import numpy as np
from setproctitle import *

from LiDAR_Detection.sfa3d.visualization import dataloader, data_utils, transformation, evaluation_utils, visualization_utils
from LiDAR_Detection.sfa3d.model import model_utils
from LiDAR_Detection.sfa3d.utils import misc, torch_utils
from LiDAR_Detection.sfa3d.config import test_config as cnf

class test():
    saved_fn = 'fpn_resnet_18'
    a = 'fpn_resnet_18'
    arch = 'fpn_resnet_18'
    pretrained_path = 'LiDAR_Detection/model/sfa3d/Model_fpn_resnet_18_epoch_300.pth'
    K = 50
    no_cuda = 'store_true'
    gpu_idx = 0
    num_samples = None
    num_workers = 1
    batch_size = 1
    peak_thresh = 0.2
    save_test_output = 'store_true'
    output_format = 'image'
    output_video_fn = 'out_fpn_resnet_18'
    output_width = 608

    pin_memory = True
    distributed = False  # For testing on 1 GPU only

    # configs.input_size = (1000, 1000)
    input_size = (608, 608)
    hm_size = (152, 152)
    down_ratio = 4
    max_objects = 50

    imagenet_pretrained = False
    head_conv = 64
    num_classes =8
    num_center_offset = 2
    num_z = 1
    num_dim = 3
    num_direction = 2  # sin, cos
    heads = {
        'hm_cen': num_classes,
        'cen_offset': num_center_offset,
        'direction': num_direction,
        'z_coor': num_z,
        'dim': num_dim
    }
    num_input_features = 4
    root_dir = './'
    dataset_dir = os.path.join(root_dir, 'Dataset')


if __name__ == '__main__':
    setproctitle('Lidar Detection')
    cnt=0
    configs=test()
    model = model_utils.create_model(configs)
    print('\n\n' + '-*=' * 30 + '\n\n')
    assert os.path.isfile(configs.pretrained_path), "No file at {}".format(configs.pretrained_path)
    model.load_state_dict(torch.load(configs.pretrained_path, map_location='cpu'))
    print('Loaded weights from {}\n'.format(configs.pretrained_path))

    test.device = torch.device('cpu' if configs.no_cuda else 'cuda:{}'.format(configs.gpu_idx))
    model = model.to(device=configs.device)

    out_cap = None

    model.eval()

    test_dataloader = dataloader.create_test_dataloader(configs)
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(test_dataloader):

            metadatas, bev_maps, img_rgbs = batch_data
            input_bev_maps = bev_maps.to(configs.device, non_blocking=True).float()
            t1 = misc.time_synchronized()
            outputs = model(input_bev_maps)
            outputs['hm_cen'] = torch_utils._sigmoid(outputs['hm_cen'])
            outputs['cen_offset'] = torch_utils._sigmoid(outputs['cen_offset'])
            # detections size (batch_size, K, 10)
            detections = evaluation_utils.decode(outputs['hm_cen'], outputs['cen_offset'], outputs['direction'], outputs['z_coor'],
                                outputs['dim'], K=configs.K)
            detections = detections.cpu().numpy().astype(np.float32)
            detections = evaluation_utils.post_processing(detections, configs.num_classes, configs.down_ratio, configs.peak_thresh)
            t2 = misc.time_synchronized()

            detections = detections[0]  # only first batch
            # Draw prediction in the image
            bev_map = (bev_maps.squeeze().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            bev_map = cv2.resize(bev_map, (cnf.BEV_WIDTH, cnf.BEV_HEIGHT))

            bev_map = evaluation_utils.draw_predictions(bev_map, detections.copy(), configs.num_classes)

            # Rotate the bev_map
            bev_map = cv2.rotate(bev_map, cv2.ROTATE_180)
            # bev_map = cv2.rotate(bev_map, cv2.ROTATE_90_CLOCKWISE)

            img_path = metadatas['img_path'][0]
            img_rgb = img_rgbs[0].numpy()
            img_rgb = cv2.resize(img_rgb, (img_rgb.shape[1], img_rgb.shape[0]))
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            # img_bgr = cv2.rotate(img_bgr, cv2.ROTATE_90_CLOCKWISE)
            calib = data_utils.Calibration(img_path.replace(".png", ".txt").replace("image", "calib"))
            kitti_dets = evaluation_utils.convert_det_to_real_values(detections)
            if len(kitti_dets) > 0:
                kitti_dets[:, 1:] = transformation.lidar_to_camera_box(kitti_dets[:, 1:], calib.V2C, calib.R0, calib.P2)
                img_bgr = visualization_utils.show_rgb_image_with_boxes(img_bgr, kitti_dets, calib)

            out_img = visualization_utils.merge_rgb_to_bev(img_bgr, bev_map, output_width=configs.output_width)


            print('\tDone testing the {}th sample, time: {:.1f}ms, speed {:.2f}FPS'.format(batch_idx, (t2 - t1) * 1000,
                                                                                           1 / (t2 - t1)))
            test_folder = "/mnt/home/jo/Facility_Damage_Detection/Dataset/test_dataset/lidar"
            output_folder = "/mnt/home/jo/Facility_Damage_Detection/Output/LiDAR_Detection/"
            f_name=[]
            file_list= sorted(os.listdir(test_folder))
            file_name=[len(file_list)+1]
            for file_names in file_list:
                file_name = os.path.splitext(file_names)[0]
                f_name.append(file_name)
            out_path = os.path.join(output_folder, f_name[cnt] + ".jpg")
            cv2.imwrite(out_path, out_img)
            cnt+=1
            if cv2.waitKey(1) & 0xFF == 27:
                break
    if out_cap:
        out_cap.release()
    cv2.destroyAllWindows()