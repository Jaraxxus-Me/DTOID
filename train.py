import argparse
import os
import time
import logging
import numpy as np
import pickle as pkl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from parser.train_parser import train_parser
from datasets.owid import OWIDDataset
from models.network import Network
from tqdm import tqdm
from train_helper import *
from torch.utils.tensorboard import SummaryWriter
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.functional")

parser = train_parser()
args = parser.parse_args()

def gettime():
    # get GMT time in string
    return time.strftime("%Y%m%d%H%M%S", time.gmtime())

def main():
    if not os.path.isdir(args.savepath):
        os.makedirs(args.savepath)
    # args.savepath = args.savepath+f'/test_re10k_{gettime()}'
    log_file = os.path.join(args.savepath, 'training.log')
    logging.basicConfig(filename=log_file, filemode='w', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    writer = SummaryWriter(log_dir=args.savepath)

    for key, value in sorted(vars(args).items()):
        logging.info(str(key) + ': ' + str(value))
    
    # train dataset
    train_dataset = OWIDDataset(args)
    TrainLoader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True, num_workers=args.worker)
    # val dataset
    val_dataset = OWIDDataset(args, train=False)
    ValidLoader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=args.worker)

    # model
    # Load Network
    model = Network().cuda()

    if args.resume:
        if os.path.isfile(args.resume):
            logging.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage)
            model.load_state_dict(checkpoint["state_dict"])
            model = model.cuda()
            logging.info("=> loaded checkpoint '{}'".format(args.resume))
        else:
            model = model.cuda()
            logging.info("=> No checkpoint found at '{}'".format(args.resume))
            logging.info("=> Will start from scratch.")
    else:
        model = model.cuda()
        logging.info('=> No checkpoint file. Start from scratch.')

    all_param = list(model.parameters())
    optimizer = torch.optim.Adam(all_param, lr=1e-4, weight_decay=1e-6, amsgrad=True)


    start_full_time = time.time()

    for epoch in range(args.epochs):
        logging.info('This is {}-th epoch'.format(epoch))
        train(TrainLoader, ValidLoader,
                model,
                optimizer, logging, epoch, writer)

def train(TrainLoader, ValidLoader, model, optimizer, log, epoch, writer):
    start_time = time.time()
    _loss = AverageMeter()
    n_b = len(TrainLoader)
    torch.cuda.synchronize()
    b_s = time.perf_counter()
    progress_bar = tqdm(TrainLoader, desc="Epoch {}".format(epoch), dynamic_ncols=True)
    for b_i, data in enumerate(progress_bar):
        model.train()
        n_iter = b_i + n_b * epoch
        # lr adjustment
        adjust_lr(args, optimizer, epoch)
        # template
        temp = data[0].cuda()
        mask = data[1].cuda()
        template = temp
        template_mask = mask
        global_template = temp[:, 0]
        global_template_mask = mask[:, 0]
        # query image
        query = data[2].cuda()
        ann_info = data[3]
        gt_boxes = ann_info['bboxes'].cuda()
        gt_masks = ann_info['masks'][0].cuda().to(torch.float32)
        gt_heat = ann_info['heat_maps'][0].cuda()
        classifications, regression, anchors, heat_map, segmentation = model(
            query, template, template_mask, global_template, global_template_mask)
        # detection loss
        anchors = anchors.repeat(args.bs, 1, 1)
        anchor_target = generate_anchor_target(anchors, gt_boxes)
        loss_cls, loss_reg = focal_l1_loss(classifications, regression, anchor_target[0], anchor_target[1], anchor_target[0])
        # segmentation and center loss
        loss_heat = F.l1_loss(heat_map.squeeze(), gt_heat, reduction='mean')
        loss_seg = F.binary_cross_entropy_with_logits(segmentation.squeeze().view(-1), gt_masks.view(-1), reduction='mean')
        
        loss = loss_cls + loss_reg + 20*loss_heat + 20*loss_seg

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _loss.update(loss.item())

        batch_time = time.perf_counter() - b_s
        b_s = time.perf_counter()

        writer.add_scalar('Loss (Avg)', loss, n_iter)
        writer.add_scalar('Loss (Cls)', loss_cls, n_iter)
        writer.add_scalar('Loss (Reg)', loss_reg, n_iter)
        writer.add_scalar('Loss (Heat)', loss_heat, n_iter)
        writer.add_scalar('Loss (Seg)', loss_seg, n_iter)
        # Update the progress bar
        progress_bar.set_postfix({"Loss_avg": loss.item(), "batch_time": batch_time})

        if n_iter > 0 and n_iter % args.valid_freq == 0:
            with torch.no_grad():
                ar = eval_detection(ValidLoader, model, n_iter, writer)

            log.info("Saving new checkpoint.")
            savefilename = args.savepath + '/checkpoint.tar'
            save_checkpoint(model, optimizer, savefilename)
            global cur_max_psnr

    final_loss = _loss.avg
    epoch_time = time.time() - start_time
    logging.info("Epoch {} completed in {:.2f} seconds. Final loss: {:.4f}".format(epoch, epoch_time, final_loss))

if __name__ == '__main__':
    main()