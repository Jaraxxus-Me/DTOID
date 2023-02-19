from torch.utils.data import Dataset
import cv2
from copy import copy
import numpy as np
import torch
import os
from PIL import Image
from pycocotools.coco import COCO
from torchvision.transforms import transforms
import scipy

IMG_EXT = ['png', 'jpg']

normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

IMG_SIZE = (512, 512)
TEMPLATE_SIZE = 124

PREPROCESS = [transforms.Compose([transforms.Resize(IMG_SIZE[0]), transforms.ToTensor(), normalize]),
              transforms.Compose([transforms.Resize(TEMPLATE_SIZE)]),
              transforms.Compose([transforms.Resize(TEMPLATE_SIZE)]),
              transforms.Compose([transforms.ToTensor(), normalize]),
              transforms.Compose([transforms.ToTensor()])
              ]

class OWIDDataset(Dataset):
    def __init__(self, args, train=True):
        super().__init__()
        # query
        if train:
            anno = os.path.join(args.data_path, 'train_annotations.json')
            self.coco = COCO(anno)
            img_scale = 50000
            ins_scale = 180000
        else:
            anno = os.path.join(args.data_path, 'val_annotations.json')
            self.coco = COCO(anno)
            img_scale = 2000
            ins_scale = 2000
        self.img_ids = np.load(anno[:-5]+"_0.npy")
        self.img_ids = self.img_ids[:img_scale]
        self.obj_ids = self.coco.getCatIds()
        self.data_infos = []

        for i in self.img_ids:
            info = self.coco.loadImgs([i])[0]
            info['filename'] = info['file_name']
            anns = self.coco.getAnnIds(info['id'])
            for i, ann in enumerate(self.coco.loadAnns(anns)):
                local_info = copy(info)
                local_info['ann_id'] = i
                local_info['obj_id'] = ann['category_id']
                assert local_info['obj_id'] in self.obj_ids, 'Anno_{} has object that do not have P1 video'.format(ann['id'])
                self.data_infos.append(local_info)
        self.data_infos = self.data_infos[:ins_scale]
        print('Dataset scale (before filtering):\n Images:' + str(len(self.img_ids)) + '\n Instances:' +
                str(len(self.data_infos)))
        self.img_path = args.data_path
        # support
        self.p1_path = args.support_path
        for obj_id in self.obj_ids:
            pre_processed_npy = os.path.join(self.p1_path, '{}'.format(obj_id), 'info_dtoid.npz')
            if not os.path.exists(pre_processed_npy):
                print('Pre-constructing Support ID {}/{}'.format(obj_id, len(self.obj_ids)))
                support = []
                rgb_path = os.path.join(self.p1_path, '{}'.format(obj_id), 'rgb')
                for i in range(40):
                    rgb_img_path = os.path.join(rgb_path, "{:06d}.jpg".format(i))
                    mask_path = rgb_img_path.replace('rgb', 'mask')
                    template_im = cv2.imread(rgb_img_path)[:, :, ::-1]
                    template = Image.fromarray(template_im)
                    template_mask = cv2.imread(mask_path)[:, :, 0]
                    template_mask = Image.fromarray(template_mask)
                    # preprocess and concatenate
                    template = np.asarray(PREPROCESS[1](template))
                    template_mask = np.asarray(PREPROCESS[2](template_mask))[:,:,np.newaxis]
                    template = np.concatenate([template, template_mask], axis=2)
                    support.append(template)
                support = np.stack(support, axis=0)
                np.savez(pre_processed_npy, support=support)
        
    def __len__(self):
        return len(self.data_infos)
    
    def __getitem__(self, idx):
        # query img
        img_info = self.data_infos[idx]
        img_path = os.path.join(self.img_path, img_info['filename'])
        frame = cv2.imread(img_path)
        img = Image.fromarray(frame[:, :, ::-1])
        img = PREPROCESS[0](img)
        obj_id = img_info['obj_id']
        ann_info = self.get_ann_info(idx)
        # return support and query
        support_info = np.load(os.path.join(self.p1_path, '{}'.format(obj_id), 'info_dtoid.npz'))['support']
        temp_rgbs = support_info[:, :, :, :3]
        temp_masks = support_info[:, :, :, -1]
        # process
        rgb = torch.zeros(temp_rgbs.shape[0], temp_rgbs.shape[3], temp_rgbs.shape[1], temp_rgbs.shape[2])
        mask = torch.zeros(temp_masks.shape[0], 1, temp_masks.shape[1], temp_masks.shape[2])
        for i in range(40):
            temp_rgb = Image.fromarray(temp_rgbs[i])
            temp_mask = Image.fromarray(temp_masks[i])
            temp_rgb = PREPROCESS[3](temp_rgb)
            temp_mask = PREPROCESS[4](temp_mask)
            rgb[i] = temp_rgb
            mask[i] = temp_mask
        
        return rgb, mask, img, ann_info

    def get_ann_info(self, idx):
        """Get COCO annotation by index.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Annotation info of specified index.
        """

        img_id = self.data_infos[idx]['id']
        ann_ids = self.coco.getAnnIds([img_id])
        ann_info = self.coco.loadAnns(ann_ids)
        return self._parse_ann_info(self.data_infos[idx], [ann_info[self.data_infos[idx]['ann_id']]])

    def _parse_ann_info(self, img_info, ann_info):
        """Parse bbox and mask annotation.

        Args:
            ann_info (list[dict]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,\
                labels, masks, seg_map. "masks" are raw annotations and not \
                decoded into binary masks.
        """
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_masks_ann = []
        heat_maps = []
        for i, ann in enumerate(ann_info):
            assert len(ann_info) == 1
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]
            if ann.get('iscrowd', False):
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(0)
                gt_masks_ann.append(self.coco.annToMask(ann))
                heat_maps.append(self.generate_heatmap(bbox))

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            bboxes_ignore=gt_bboxes_ignore,
            masks=gt_masks_ann,
            heat_maps=heat_maps)

        return ann

    def generate_heatmap(self, gt_box, original_image_size=(512, 512), feature_map_size=(31, 31)):
        # Calculate the center coordinates and width and height of the ground truth box in the original image
        x_center = (gt_box[0] + gt_box[2]) / 2.0
        y_center = (gt_box[1] + gt_box[3]) / 2.0
        width = gt_box[2] - gt_box[0]
        height = gt_box[3] - gt_box[1]

        # Transform the center coordinates and width and height to the corresponding feature map coordinates
        x_center_fm = x_center * feature_map_size[1] / original_image_size[1]
        y_center_fm = y_center * feature_map_size[0] / original_image_size[0]
        width_fm = width * feature_map_size[1] / original_image_size[1]
        height_fm = height * feature_map_size[0] / original_image_size[0]

        # Create a 2D Gaussian kernel centered at the center coordinates with a standard deviation proportional to the width and height
        x, y = np.meshgrid(np.arange(feature_map_size[1]), np.arange(feature_map_size[0]))
        d = np.sqrt((x - x_center_fm)**2 + (y - y_center_fm)**2)
        sigma = max(width_fm, height_fm) / 6.0
        kernel = scipy.stats.norm.pdf(d, 0, sigma)

        # Normalize the kernel to sum to 1 and return it as a PyTorch tensor
        kernel /= kernel.max()
        kernel = torch.from_numpy(kernel).float()

        return kernel