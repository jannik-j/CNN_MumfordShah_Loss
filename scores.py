import torch


def iou(out, true, smooth=1):
    true_ = torch.zeros_like(out)
    for c in range(out.shape[0]):
        true_[c] = (true == c).float()
    out_ = out.clone().detach()
    intersection = torch.sum(torch.abs(out_*true_), (1, 2))
    union = torch.sum(out_, (1, 2)) + torch.sum(true_, (1, 2)) - intersection
    iou = torch.mean((intersection+smooth) / (union+smooth), 0)
    return iou.item()


def dice(out, true, smooth=1):
    true_ = torch.zeros_like(out)
    for c in range(out.shape[0]):
        true_[c] = (true == c).float()
    out_ = out.clone().detach()
    intersection = torch.sum(torch.abs(out_*true_), (1, 2))
    union = torch.sum(out_, (1, 2)) + torch.sum(true_, (1, 2))
    dice = torch.mean((2.*intersection + smooth) / (union+smooth), 0)
    return dice.item()


def iou_tumoronly(out, true):
    true_ = true.clone().detach()
    out_ = out.clone().detach()
    intersection = torch.sum(torch.abs(out_*true_), (0, 1))
    union = torch.sum(out_, (0, 1)) + torch.sum(true_, (0, 1)) - intersection
    iou = intersection/union
    return iou.item()


def dice_tumoronly(out, true):
    true_ = true.clone().detach()
    out_ = out.clone().detach()
    intersection = torch.sum(torch.abs(out_*true_), (0, 1))
    union = torch.sum(out_, (0, 1)) + torch.sum(true_, (0, 1))
    dice = (2.*intersection) / union
    return dice.item()
