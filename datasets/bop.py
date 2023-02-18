from torch.utils.data import Dataset
import cv2
import numpy as np
import torch
import os
from PIL import Image
from pycocotools.coco import COCO
from torchvision.transforms import transforms

IMG_EXT = ['png', 'jpg']

normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

IMG_SIZE = (480, 640)
HEATMAP_SIZE = (29, 39)
TEMPLATE_SIZE = 124

PREPROCESS = [transforms.Compose([transforms.Resize(IMG_SIZE[0]), transforms.ToTensor(), normalize]),
              transforms.Compose([transforms.Resize(TEMPLATE_SIZE), transforms.ToTensor(), normalize]),
              transforms.Compose([transforms.Resize(TEMPLATE_SIZE), transforms.ToTensor()]),
              transforms.Compose([transforms.Resize(TEMPLATE_SIZE), transforms.ToTensor(), normalize]),
              transforms.Compose([transforms.Resize(TEMPLATE_SIZE), transforms.ToTensor()])
              ]

class BOPDataset(Dataset):
    def __init__(self, args):
        super().__init__()
        # query
        test_anno = os.path.join(args.data_path, 'scene_gt_coco_all.json')
        self.coco = COCO(test_anno)
        self.img_ids = self.coco.getImgIds()
        self.cat_ids = self.coco.getCatIds()
        self.data_infos = []
        for i in self.img_ids:
            info = self.coco.loadImgs([i])[0]
            anns = self.coco.getAnnIds(info['id'])
            for i, ann in enumerate(self.coco.loadAnns(anns)):
                ann['filename'] = info['file_name']
                self.data_infos.append(ann)
        self.img_path = args.data_path
        # support
        self.support = {}
        self.p1_path = args.support_path
        for obj_id in self.cat_ids:
            support = []
            rgb_path = os.path.join(self.p1_path, 'obj_{:06d}'.format(obj_id), 'rgb')
            for i in range(160):
                rgb_img_path = os.path.join(rgb_path, "{:06d}.jpg".format(i))
                mask_path = rgb_img_path.replace('rgb', 'mask')
                template_im = cv2.imread(rgb_img_path)[:, :, ::-1]
                template = Image.fromarray(template_im)
                template_mask = cv2.imread(mask_path)[:, :, 0]
                template_mask = Image.fromarray(template_mask)
                # preprocess and concatenate
                template = PREPROCESS[1](template)
                template_mask = PREPROCESS[2](template_mask)
                template = torch.cat([template, template_mask], dim=0)
                support.append(template)
            self.support[obj_id] = support
        
    def __len__(self):
        return len(self.data_infos)
    
    def __getitem__(self, idx):
        # query img
        img_info = self.data_infos[idx]
        img_path = os.path.join(self.img_path, img_info['filename'])
        frame = cv2.imread(img_path)
        img_h, img_w, img_c = frame.shape
        img = Image.fromarray(frame[:, :, ::-1])
        img = PREPROCESS[0](img)
        network_h = img.size(1)
        network_w = img.size(2)
        img_id = img_info['image_id']
        obj_id = img_info['category_id']
        # return support and query
        support_info = {'id': obj_id, 'data':self.support[obj_id]}
        query_info = {'img': img, 'img_h': img_h, 'img_w': img_w,
                       'img_id': img_id,'network_h': network_h, 'network_w': network_w
        }
        return support_info, query_info