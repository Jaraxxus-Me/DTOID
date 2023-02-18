import os
import cv2
from tqdm import tqdm

dtoid_temp = 'templates'
air_temp = 'data/BOP/lmo/test_video_dtoid'
obj_ids = [1, 5, 6, 8, 9, 10, 11, 12]

for obj_id in tqdm(obj_ids):
    # tar
    tar_path = os.path.join(air_temp, 'obj_{:06d}'.format(obj_id))
    tar_path_rgb = os.path.join(tar_path, 'rgb')
    tar_path_mask = os.path.join(tar_path, 'mask')
    tar_path_depth = os.path.join(tar_path, 'depth')
    if not os.path.isdir(tar_path_rgb):
        os.makedirs(tar_path_rgb)
        os.makedirs(tar_path_mask)
        os.makedirs(tar_path_depth)
    # ori
    ori_path = os.path.join(dtoid_temp, 'hinterstoisser_{:02d}'.format(obj_id))
    ori_rgb = []
    ori_mask = []
    ori_depth = []
    for path in os.listdir(ori_path):
        if '_a' in path:
            ori_rgb.append(os.path.join(ori_path, path))
        elif '_m' in path:
            ori_mask.append(os.path.join(ori_path, path))
        else:
            ori_depth.append(os.path.join(ori_path, path))
    # re-write
    for ori_img in ori_rgb:
        # rgb
        img_rgb = cv2.imread(ori_img)
        tar_img_rgb = os.path.join(tar_path_rgb, ori_img.split('/')[-1][:6]+'.jpg')
        cv2.imwrite(tar_img_rgb, img_rgb)
        # mask
        img_mask = cv2.imread(ori_img.replace('_a', '_m'))
        tar_img_mask = os.path.join(tar_path_mask, ori_img.split('/')[-1][:6]+'.jpg')
        cv2.imwrite(tar_img_mask, img_mask)
        # rgb
        img_depth = cv2.imread(ori_img.replace('_a', '_d'))
        tar_img_depth = os.path.join(tar_path_depth, ori_img.split('/')[-1][:6]+'.png')
        cv2.imwrite(tar_img_depth, img_depth)
    
