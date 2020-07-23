import time
import os
import numpy as np
from options.train_options import TrainOptions
from LiTS_getDatabase_unet import DataProvider_LiTS
from models.models import create_model
from util.visualizer import Visualizer
from collections import OrderedDict
from math import *
from util import html
import scipy.io as sio
import torch
import matplotlib.pyplot as plt


def iou(out, true, smooth=1):
    true_ = torch.stack((true.clone().detach(), true.clone().detach()))
    true_[0] = 1 - true_[0]
    out_ = out.clone().detach()
    intersection = torch.sum(torch.abs(out_*true_), (1, 2))
    union = torch.sum(out_, (1, 2)) + torch.sum(true_, (1, 2)) - intersection
    iou = torch.mean((intersection+smooth) / (union+smooth), 0)
    return iou.item()


def dice(out, true, smooth=1):
    true_ = torch.stack((true.clone().detach(), true.clone().detach()))
    true_[0] = 1 - true_[0]
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


def getBothVisuals(visuals_liver, visuals_tumor):
    visuals = OrderedDict([('Slice0', visuals_liver['real_A1']), ('Slice1', visuals_liver['real_A2']),
                           ('Slice2', visuals_liver['real_A3']), ('Label_Liver', visuals_liver['real_B2']),
                           ('Seg_Background_Liver', visuals_liver['fake_B0']), ('Seg_Liver', visuals_liver['fake_B1']),
                           ('Liver0', visuals_tumor['real_A1']), ('Liver1', visuals_tumor['real_A2']),
                           ('Liver2', visuals_tumor['real_A2']), ('Label_Tumor', visuals_tumor['real_B2']),
                           ('Seg_Background_Tumor', visuals_tumor['fake_B0']), ('Seg_Tumor', visuals_tumor['fake_B1'])
                           ])
    return visuals


opt = TrainOptions().parse()
opt.isTrain = False
input_min       = 0
input_max       = 400

if opt.segType != 'both':
    model = create_model(opt)
    visualizer = Visualizer(opt)

    data_train = DataProvider_LiTS(opt.inputSize, opt.fineSize, opt.segType, opt.semi_rate, opt.input_nc,
                                   opt.dataroot, a_min=input_min, a_max=input_max, mode="test")
    dataset_size = data_train.n_data
    print('#testing images = %d' % dataset_size)
    dice_cumul, iou_cumul = .0, .0

    for step in range(1, dataset_size):
        batch_x, batch_y, path = data_train(opt.batchSize)
        data = {'A': batch_x, 'A_paths': path, 'B': batch_y, 'B_paths': path}
        model.set_input(data)
        model.test()

        # Falls nötig #
        model.fake_B2[0, 0] = (model.fake_B2[0, 0] > 0.6).float()
        model.fake_B2[0, 1] = (model.fake_B2[0, 1] > 0.4).float()
        #################
        # model.fake_B2[0, 0] = (model.fake_B2[0, 0] > 0.5).float()
        # model.fake_B2[0, 1] = (model.fake_B2[0, 1] > 0.5).float()

        score_dice = dice(model.fake_B2[0], batch_y[0, 0])
        score_iou = iou(model.fake_B2[0], batch_y[0, 0])
        dice_cumul += score_dice
        iou_cumul += score_iou

        if step % opt.display_step == 0:
            print("Step %d | Dice: %f, IoU: %f" % (step, score_dice, score_iou))
            visualizer.display_current_results(model.get_current_visuals(), 1, False)

    dice_complete = dice_cumul / dataset_size
    iou_complete = iou_cumul / dataset_size
    print("--------------------------------------------------------")
    print('End score:')
    print("Dice: %f, IoU: %f" % (dice_complete, iou_complete))

else:
    data_train = DataProvider_LiTS(opt.inputSize, opt.fineSize, opt.segType, opt.semi_rate, opt.input_nc,
                                   opt.dataroot, a_min=input_min, a_max=input_max, mode="test")
    opt.segType = 'tumor'
    model_tumor = create_model(opt)
    visualizer = Visualizer(opt)

    opt.segType = 'liver'
    opt.checkpoints_dir = './checkpoints/2020-06-04_LiTS_Liver_semi__semi=10'
    # opt.which_epoch = 10            # Für Netzwerke aus unterschiedlichen Epochen
    model_liver = create_model(opt)
    dataset_size = data_train.n_data
    print('#testing images = %d' % dataset_size)
    dice_liver_cumul, iou_liver_cumul = .0, .0
    dice_tumor_cumul, iou_tumor_cumul = .0, .0

    for step in range(dataset_size):
        batch_x, batch_y, path = data_train(opt.batchSize)
        true_liver = batch_y.clone()
        true_liver = (true_liver > 0).float()
        data = {'A': batch_x, 'A_paths': path, 'B': true_liver, 'B_paths': path}
        model_liver.set_input(data)
        model_liver.test()
        model_liver.fake_B2[0, 0] = (model_liver.fake_B2[0, 0] > 0.5).float()
        model_liver.fake_B2[0, 1] = (model_liver.fake_B2[0, 1] > 0.5).float()

        score_dice_liver = dice(model_liver.fake_B2[0], true_liver[0, 0])
        score_iou_liver = iou(model_liver.fake_B2[0], true_liver[0, 0])
        dice_liver_cumul += score_dice_liver
        iou_liver_cumul += score_iou_liver

        for ich in range(opt.input_nc):
            batch_x[0, ich] = batch_x[0, ich] * model_liver.fake_B2[0, 1]
            batch_x[0, ich] -= torch.min(batch_x[0, ich])
            batch_x[0, ich] /= torch.max(batch_x[0, ich])

        true_tumor = batch_y.clone()
        true_tumor = (true_tumor == 2).float()
        data = {'A': batch_x, 'A_paths': path, 'B': true_tumor, 'B_paths': path}
        model_tumor.set_input(data)
        model_tumor.test()
        # FÜR MSLOSS MIT SEMIRATE 10 #
        model_tumor.fake_B2[0, 0] = (model_tumor.fake_B2[0, 0] > 0.6).float()
        model_tumor.fake_B2[0, 1] = (model_tumor.fake_B2[0, 1] > 0.4).float()
        ###################
        # model_tumor.fake_B2[0, 0] = (model_tumor.fake_B2[0, 0] > 0.5).float()
        # model_tumor.fake_B2[0, 1] = (model_tumor.fake_B2[0, 1] > 0.5).float()

        score_dice_tumor = dice(model_tumor.fake_B2[0], true_tumor[0, 0])
        # score_dice_tumor = dice_tumoronly(model_tumor.fake_B2[0, 1], true_tumor[0, 0])
        score_iou_tumor = iou(model_tumor.fake_B2[0], true_tumor[0, 0])
        # score_iou_tumor = iou_tumoronly(model_tumor.fake_B2[0, 1], true_tumor[0, 0])
        dice_tumor_cumul += score_dice_tumor
        iou_tumor_cumul += score_iou_tumor

        if step % opt.display_step == 0:
            visuals_liver = model_liver.get_current_visuals()
            visuals_tumor = model_tumor.get_current_visuals()
            # print("Step %d | %s" % (step, path))
            print("Step %d | Dice_liver: %f, IoU_liver: %f" % (step, score_dice_liver, score_iou_liver))
            print("Step %d | Dice_tumor: %f, IoU_tumor: %f" % (step, score_dice_tumor, score_iou_tumor))
            visualizer.display_current_results(getBothVisuals(visuals_liver, visuals_tumor), 1, False)

    dice_liver_complete = dice_liver_cumul / dataset_size
    iou_liver_complete = iou_liver_cumul / dataset_size
    dice_tumor_complete = dice_tumor_cumul / dataset_size
    iou_tumor_complete = iou_tumor_cumul / dataset_size
    print("--------------------------------------------------------")
    print('End score:')
    print("LIVER | Dice: %f, IoU: %f" % (dice_liver_complete, iou_liver_complete))
    print("TUMOR | Dice: %f, IoU: %f" % (dice_tumor_complete, iou_tumor_complete))
