import argparse
import os
import time
import logging
import numpy as np
import pickle as pkl
import torch
from torch.utils.data import DataLoader
from parser.test_bop_parser import test_parser
from importlib.machinery import SourceFileLoader
import torchvision.io as io
from datasets.bop import BOPDataset
from models.network import Network
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.functional")

parser = test_parser()
args = parser.parse_args()

def gettime():
    # get GMT time in string
    return time.strftime("%Y%m%d%H%M%S", time.gmtime())

def main():
    if not os.path.isdir(args.savepath):
        os.makedirs(args.savepath)
    # args.savepath = args.savepath+f'/test_re10k_{gettime()}'
    log = logging.getLogger('Log')

    for key, value in sorted(vars(args).items()):
        log.info(str(key) + ': ' + str(value))
    
    # dataset
    dataset = BOPDataset(args)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # model
    # Load Network
    model = Network()
    model.eval()

    if args.resume:
        if os.path.isfile(args.resume):
            log.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage)
            model.load_state_dict(checkpoint["state_dict"])
            model = model.cuda()
            log.info("=> loaded checkpoint '{}'".format(args.resume))
        else:
            model = model.cuda()
            log.info("=> No checkpoint found at '{}'".format(args.resume))
            log.info("=> Will start from scratch.")
    else:
        model = model.cuda()
        log.info('=> No checkpoint file. Start from scratch.')

    start_full_time = time.time()
    with torch.no_grad():
        log.info('start testing.')
        predictions = test(dataloader, model, log)
    log.info('full testing time = {:.2f} Minutes'.format((time.time() - start_full_time) / 60))
    pred_save = os.path.join(args.savepath, 'res.pkl')
    with open(pred_save, 'wb') as f:
        pkl.dump(predictions, f)

def test(dataloader, model, log):
    predictions = {}
    template_features = {}
    for b_i, data in tqdm(enumerate(dataloader)):
        model.eval()
        # template
        template = data[0]
        obj_id = int(template['id'])
        if obj_id not in template_features.keys():
            log.info('Generating template feature for OBJ {}.'.format(obj_id))
            template_imgs = template['data']
            iteration = 0
            # features for all templates (240)
            template_list = []
            template_global_list = []
            batch_size = 10
            temp_batch_local = []
            iteration = 0
            for template_img in template_imgs:
                template_img = template_img.cuda()
                template_feature = model.compute_template_local(template_img)
                # Create mini-batches of templates
                if iteration == 0:
                    temp_batch_local = template_feature

                    template_feature_global = model.compute_template_global(template_img)
                    template_global_list.append(template_feature_global)

                elif iteration % (batch_size) == 0:
                    template_list.append(temp_batch_local)
                    temp_batch_local = template_feature

                elif iteration == (len(template_imgs) - 1):
                    temp_batch_local = torch.cat([temp_batch_local, template_feature], dim=0)
                    template_list.append(temp_batch_local)

                else:
                    temp_batch_local= torch.cat([temp_batch_local, template_feature], dim=0)

                iteration += 1
                
            template_features[obj_id] = {'template_list': template_list, 'template_global_list': template_global_list}
        else:
            template_list = template_features[obj_id]['template_list']
            template_global_list = template_features[obj_id]['template_global_list']

        # query image
        query = data[1]
        # img_h, img_w = int(query['img_h']), int(query['img_w'])
        # network_h, network_w = int(query['network_h']), int(query['network_w'])
        img = query['img'].cuda()
        top_k_num = 500
        top_k_scores, top_k_bboxes, top_k_template_ids = model.forward_all_templates(
            img, template_list, template_global_list, topk=top_k_num)
        pred_res = torch.cat([top_k_bboxes, top_k_scores.unsqueeze(1)], dim=1)
        pred_res_np = pred_res.cpu().numpy()
        # save res
        # if len(pred_bbox_np) > 0:
        #     # x1, y1, x2, y2 = pred_bbox_np
        #     # temp_score = pred_scores_np
        #     # x1 = int(x1 / network_w * img_w)
        #     # x2 = int(x2 / network_w * img_w)
        #     # y1 = int(y1 / network_h * img_h)
        #     # y2 = int(y2 / network_h * img_h)
        #     res = np.concatenate([x1, y1, x2, y2, temp_score], axis=1)
        predictions[int(query['img_id'])] = pred_res_np

    return predictions

if __name__ == '__main__':
    main()