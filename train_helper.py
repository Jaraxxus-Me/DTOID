import torch
import torch.nn.functional as F
from torchvision import ops
import numpy as np
import scipy
from tqdm import tqdm

def adjust_lr(args, optimizer, epoch):
    if epoch <= 20:
        lr = args.lr
    elif epoch <= 40:
        lr = args.lr * 0.1
    else:
        lr = args.lr * 0.01

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def save_checkpoint(model, optimizer, savefilename):
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, savefilename)

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def generate_anchor_target(anchors, gt_boxes, pos_iou_thr=0.5, neg_iou_thr=0.4):
    """
    Generates the anchor classification and regression targets.

    Args:
        anchors: a tensor of shape (num_anchors, 4) representing the anchor boxes
        gt_boxes: a tensor of shape (num_boxes, 4) representing the ground truth boxes
        pos_iou_thr: the IoU threshold for a positive anchor
        neg_iou_thr: the IoU threshold for a negative anchor
        device: the device on which to create the target tensors

    Returns:
        labels: a tensor of shape (batch_size, num_anchors) representing the class labels for each anchor
        regression_targets: a tensor of shape (batch_size, num_anchors, 4) representing the regression targets for each anchor
    """
    batch_size = gt_boxes.size(0)
    num_anchors = anchors.size(1)

    labels = torch.zeros((batch_size, num_anchors), dtype=torch.long, device=anchors.device)
    regression_targets = torch.zeros((batch_size, num_anchors, 4), dtype=torch.float, device=anchors.device)

    for b in range(batch_size):
        ious = ops.box_iou(anchors[b], gt_boxes[b])

        max_ious, max_idxs = ious.max(dim=1)

        pos_mask = max_ious >= pos_iou_thr
        neg_mask = max_ious < neg_iou_thr
        ignored_mask = (max_ious >= neg_iou_thr) & (~pos_mask)

        labels[b][ignored_mask] = -1
        labels[b][neg_mask & ~ignored_mask] = 1

        labels[b][pos_mask] = 0

        regression_targets[b][pos_mask] = bbox_transform(anchors[b][pos_mask], gt_boxes[b][max_idxs[pos_mask]])

    return labels, regression_targets

def bbox_transform(anchors, targets):
    """
    Calculates the regression targets for the positive anchors.

    Args:
        anchors: a tensor of shape (num_anchors, 4) representing the positive anchors
        targets: a tensor of shape (num_anchors, 4) representing the ground truth boxes corresponding to the positive anchors

    Returns:
        regression_targets: a tensor of shape (num_anchors, 4) representing the regression targets for each anchor
    """
    tx = (targets[:, 0] - anchors[:, 0]) / anchors[:, 2]
    ty = (targets[:, 1] - anchors[:, 1]) / anchors[:, 3]
    tw = torch.log(targets[:, 2] / anchors[:, 2])
    th = torch.log(targets[:, 3] / anchors[:, 3])

    regression_targets = torch.stack((tx, ty, tw, th), dim=1)

    return regression_targets


def focal_l1_loss(classification_preds, regression_preds, classification_targets, regression_targets, anchor_labels, alpha=0.25, gamma=2.0, smooth_l1_sigma=3.0):
    """
    Calculates the focal L1 loss for object detection.

    Args:
        classification_preds: a tensor of shape (batch_size, num_anchors, num_classes) representing the predicted classification scores
        regression_preds: a tensor of shape (batch_size, num_anchors, 4) representing the predicted regression offsets
        classification_targets: a tensor of shape (batch_size, num_anchors) representing the target classification labels (positive, negative, or ignored)
        regression_targets: a tensor of shape (batch_size, num_anchors, 4) representing the target regression offsets for positive anchors
        anchor_labels: a tensor of shape (batch_size, num_anchors) representing the label of each anchor (positive, negative, or ignored)
        alpha: the weighting factor for the focal loss
        gamma: the focusing parameter for the focal loss
        smooth_l1_sigma: the smoothing factor for the L1 loss

    Returns:
        loss: a scalar tensor representing the focal L1 loss
    """
    # Flatten the predictions and targets to simplify the calculation
    classification_preds = classification_preds.view(-1, classification_preds.size(-1))
    regression_preds = regression_preds.view(-1, 4)
    classification_targets = classification_targets.view(-1)
    regression_targets = regression_targets.view(-1, 4)
    anchor_labels = anchor_labels.view(-1)

    # Filter out ignored anchors
    valid_mask = (anchor_labels != -1)
    classification_preds = classification_preds[valid_mask]
    regression_preds = regression_preds[valid_mask]
    classification_targets = classification_targets[valid_mask]
    regression_targets = regression_targets[valid_mask]

    pos_mask = (classification_targets == 0)

    # Calculate the classification loss
    classification_loss = F.cross_entropy(classification_preds, classification_targets, reduction='none')
    classification_loss = torch.sum(classification_loss)

    # Calculate the focal loss weighting
    # alpha_factor = torch.where(torch.eq(classification_targets, 1), torch.ones_like(classification_targets) * alpha, 1.0 - (torch.ones_like(classification_targets) * alpha))
    # focal_weight = torch.where(torch.eq(classification_targets, 1), 1.0 - classification_preds, classification_preds)
    # focal_weight = alpha_factor * torch.pow(focal_weight, gamma)

    # Calculate the regression loss
    regression_loss = F.smooth_l1_loss(regression_preds[pos_mask], regression_targets[pos_mask], reduction='mean', beta=smooth_l1_sigma)
    regression_loss = torch.sum(regression_loss)

    # Weight the losses based on anchor labels
    # positive_mask = (classification_targets == 1)
    # negative_mask = (classification_targets == 0)
    # positive_weight = focal_weight[positive_mask]
    # negative_weight = focal_weight[negative_mask]
    # regression_weight = positive_mask.float()
    # num_positive_anchors = max(1, positive_mask.sum().float())
    # num_negative_anchors = max(1, negative_mask.sum().float())
    # classification_weight = torch.cat((positive_weight, negative_weight), dim=0)
    # classification_weight = classification_weight / (num_positive_anchors + num_negative_anchors)

    # Calculate the final loss as the weighted sum of classification and regression losses
    classification_loss /= sum(valid_mask)

    return classification_loss, regression_loss

def eval_detection(ValidLoader, model, n_iter):
    model.eval()
    progress_bar = tqdm(ValidLoader, desc="Iter {}".format(n_iter), dynamic_ncols=True)
    for b_i, data in enumerate(progress_bar):
        # template
        temp = data[0].cuda()
        mask = data[1].cuda()
        template = temp
        template_mask = mask
        template_with_mask = torch.cat([template, template_mask], dim=2)
        num_templates = template_with_mask.shape[1]
        iteration = 0
        # features for all templates (240)
        template_list = []
        template_global_list = []
        batch_size = 10
        temp_batch_local = []
        iteration = 0
        for i in range(num_templates):
            template_img = template_with_mask[:, i]
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

            elif iteration == (num_templates - 1):
                temp_batch_local = torch.cat([temp_batch_local, template_feature], dim=0)
                template_list.append(temp_batch_local)

            else:
                temp_batch_local= torch.cat([temp_batch_local, template_feature], dim=0)

            iteration += 1
        # query image
        query = data[2].cuda()
        ann_info = data[3]
        gt_boxes = ann_info['bboxes'].cuda()

        top_k_num = 500
        top_k_scores, top_k_bboxes, top_k_template_ids = model.forward_all_templates(
            query, template_list, template_global_list, topk=top_k_num)
        pred_res = torch.cat([top_k_bboxes, top_k_scores.unsqueeze(1)], dim=1)
        




