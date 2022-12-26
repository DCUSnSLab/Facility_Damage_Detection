# Author: Zylo117

"""
Simple Inference Script of EfficientDet-Pytorch
"""
import time
import torch
from torch.backends import cudnn
from matplotlib import colors

from backbone import EfficientDetBackbone
import cv2
import numpy as np
import os
from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import preprocess, invert_affine, postprocess, STANDARD_COLORS, standard_to_bgr, get_index_label, plot_one_box


def display(preds, imgs, imshow=True, imwrite=False):

    for i in range(len(imgs)):
        if len(preds[i]['rois']) == 0:
            print("here", len(preds[i]['rois']))
            continue

        imgs[i] = imgs[i].copy()

        for j in range(len(preds[i]['rois'])):
            x1, y1, x2, y2 = preds[i]['rois'][j].astype(np.int)
            obj = obj_list[preds[i]['class_ids'][j]]
            # print(obj)
            score = float(preds[i]['scores'][j])
            plot_one_box(imgs[i], [x1, y1, x2, y2], label=obj,score=score,color=color_list[get_index_label(obj, obj_list)])


        if imshow:
            cv2.imshow('img', imgs[i])
            cv2.waitKey(0)

        if imwrite:
            print("here?",str(f'test/img_inferred_d{compound_coef}_this_repo_{i}.jpg'))
            cv2.imwrite(f'test/img_inferred_d{compound_coef}_this_repo_{i}.jpg', imgs[i])

# def get_img_paths(dir_path):

def get_images_path(img_dir):
    lis = os.listdir(img_dir)
    lis = [os.path.join(img_dir,x) for x in lis]
    return lis

if __name__ == '__main__':
    img_paths = get_images_path('test/imgs')
    compound_coef = 7
    force_input_size = None  # set None to use default size
    # img_path = 'test/img.png'

    anchor_ratios = [(0.7, 1.4), (1.0, 1.0), (1.5, 0.7)]
    anchor_scales = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]
    threshold = 0.2
    iou_threshold = 0.2
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    use_cuda = True
    use_float16 = False
    cudnn.fastest = True
    cudnn.benchmark = True
    obj_list = ['damage', 'surfacePeeling', 'distortion']

    color_list = standard_to_bgr(STANDARD_COLORS)

    input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
    input_size = input_sizes[compound_coef] if force_input_size is None else force_input_size
    ori_imgs, framed_imgs, framed_metas = preprocess(*img_paths, max_size=input_size)

    if use_cuda:
        x = torch.stack([torch.from_numpy(fi).cuda() for fi in framed_imgs], 0)
    else:
        x = torch.stack([torch.from_numpy(fi) for fi in framed_imgs], 0)

    x = x.to(torch.float32 if not use_float16 else torch.float16).permute(0, 3, 1, 2)

    model = EfficientDetBackbone(compound_coef=compound_coef, num_classes=len(obj_list),
                                 ratios=anchor_ratios, scales=anchor_scales)
    model.load_state_dict(torch.load(f'weights/fac.pth', map_location='cpu'))
    model.requires_grad_(False)
    model.eval()

    if use_cuda:
        model = model.cuda()
    if use_float16:
        model = model.half()

    with torch.no_grad():
        features, regression, classification, anchors = model(x)

        regressBoxes = BBoxTransform()
        clipBoxes = ClipBoxes()

        out = postprocess(x,
                          anchors, regression, classification,
                          regressBoxes, clipBoxes,
                          threshold, iou_threshold)

    out = invert_affine(framed_metas, out)
    display(out, ori_imgs, imshow=False, imwrite=True)

    print('running speed test...')
    with torch.no_grad():
        print('test1: model inferring and postprocessing')
        print('inferring image for 10 times...')
        t1 = time.time()
        for _ in range(10):
            _, regression, classification, anchors = model(x)

            out = postprocess(x,
                              anchors, regression, classification,
                              regressBoxes, clipBoxes,
                              threshold, iou_threshold)
            out = invert_affine(framed_metas, out)

        t2 = time.time()
        tact_time = (t2 - t1) / 10
        print(f'{tact_time} seconds, {1 / tact_time} FPS, @batch_size 1')

