from options.train_options import TrainOptions
from LiTS_getDatabase_unet import DataProvider_LiTS
from models.models import create_model
from util.visualizer import Visualizer
from collections import OrderedDict
import torch
from scores import dice, iou

"""
Aus der originalen Implementation
Skript zum Testen auf dem LiTS Datensatz
"""


def getBothVisuals(visuals_liver, visuals_tumor):
    """
    Hilfsfunktion, um die Ein- und Ausgaben aller beteiligten Netzwerke und
    deren Label beim Testen von zwei Netzwerken, die hintereinander ausgeführt werden
    """
    visuals = OrderedDict([('Slice0', visuals_liver['real_A1']), ('Slice1', visuals_liver['real_A2']),
                           ('Slice2', visuals_liver['real_A3']), ('Label_Liver', visuals_liver['real_B2']),
                           ('Seg_Background_Liver', visuals_liver['fake_B0']), ('Seg_Liver', visuals_liver['fake_B1']),
                           ('Liver0', visuals_tumor['real_A1']), ('Liver1', visuals_tumor['real_A2']),
                           ('Liver2', visuals_tumor['real_A2']), ('Label_Tumor', visuals_tumor['real_B2']),
                           ('Seg_Background_Tumor', visuals_tumor['fake_B0']), ('Seg_Tumor', visuals_tumor['fake_B1'])
                           ])
    return visuals

# Parsen der Kommandozeilenargumente
opt = TrainOptions().parse()
opt.isTrain = False

input_min       = 0
input_max       = 400

if opt.segType != 'both':
    """
    Falls nur ein Netzwerk getestet werden soll, wird der folgende Teil ausgeführt
    """
    # Netzwerk laden, visdom-Sitzung starten
    model = create_model(opt)
    visualizer = Visualizer(opt)

    # Daten über die Klasse DataProvider_LiTS laden
    data_train = DataProvider_LiTS(opt.inputSize, opt.fineSize, opt.segType, opt.semi_rate, opt.input_nc,
                                   opt.dataroot, a_min=input_min, a_max=input_max, mode="test")
    dataset_size = data_train.n_data
    print('#testing images = %d' % dataset_size)
    # Variablen, um den Gesamtdurchschnitt der Bewertungen zu berechnen
    dice_cumul, iou_cumul = .0, .0

    for step in range(1, dataset_size):
        # Auswertung auf den Daten im test set und Berechnung der Bewertungen

        batch_x, batch_y, path = data_train(opt.batchSize)
        data = {'A': batch_x, 'A_paths': path, 'B': batch_y, 'B_paths': path}
        model.set_input(data)
        model.test()

        # Falls nötig, wenn sonst alle Pixel dem Hintergrund zugeordnet werden #
        # model.fake_B2[0, 0] = (model.fake_B2[0, 0] > 0.6).float()
        # model.fake_B2[0, 1] = (model.fake_B2[0, 1] > 0.4).float()
        #################
        model.fake_B2[0, 0] = (model.fake_B2[0, 0] > 0.5).float()
        model.fake_B2[0, 1] = (model.fake_B2[0, 1] > 0.5).float()

        score_dice = dice(model.fake_B2[0], batch_y[0, 0])
        score_iou = iou(model.fake_B2[0], batch_y[0, 0])
        dice_cumul += score_dice
        iou_cumul += score_iou

        # Visualisierung über visdom
        if step % opt.display_step == 0:
            print("Step %d | Dice: %f, IoU: %f" % (step, score_dice, score_iou))
            visualizer.display_current_results(model.get_current_visuals(), 1, False)

    # Berechnung der Bewertungsdurchschnitte
    dice_complete = dice_cumul / dataset_size
    iou_complete = iou_cumul / dataset_size
    print("--------------------------------------------------------")
    print('End score:')
    print("Dice: %f, IoU: %f" % (dice_complete, iou_complete))

else:
    """
    Falls zwei Netzwerke getestet werden sollen, die hintereinander ausgeführt
    werden, wird der folgende Teil ausgeführt
    """

    # Daten über die Klasse DataProvider_LiTS laden
    data_train = DataProvider_LiTS(opt.inputSize, opt.fineSize, opt.segType, opt.semi_rate, opt.input_nc,
                                   opt.dataroot, a_min=input_min, a_max=input_max, mode="test")
    # Laden des Netzwerks zur Segmentierung des Tumors
    opt.segType = 'tumor'
    model_tumor = create_model(opt)
    visualizer = Visualizer(opt)

    # Laden des Netzwerks zur Segmentierung der Leber
    opt.segType = 'liver'
    opt.checkpoints_dir = './checkpoints/2020-06-04_LiTS_Liver_semi__semi=10'
    # opt.which_epoch = 10            # Für Netzwerke aus unterschiedlichen Epochen
    model_liver = create_model(opt)
    dataset_size = data_train.n_data
    print('#testing images = %d' % dataset_size)

    # Variablen zur Berechnung des Gesamtdurchschnitts der Bewertungen
    dice_liver_cumul, iou_liver_cumul = .0, .0
    dice_tumor_cumul, iou_tumor_cumul = .0, .0

    for step in range(dataset_size):
        # Auswertung des Leber-Netzwerks
        batch_x, batch_y, path = data_train(opt.batchSize)
        true_liver = batch_y.clone()
        true_liver = (true_liver > 0).float()
        data = {'A': batch_x, 'A_paths': path, 'B': true_liver, 'B_paths': path}
        model_liver.set_input(data)
        model_liver.test()
        model_liver.fake_B2[0, 0] = (model_liver.fake_B2[0, 0] > 0.5).float()
        model_liver.fake_B2[0, 1] = (model_liver.fake_B2[0, 1] > 0.5).float()

        # Berechnung der Bewertung des Leber-Netzwerks
        score_dice_liver = dice(model_liver.fake_B2[0], true_liver[0, 0])
        score_iou_liver = iou(model_liver.fake_B2[0], true_liver[0, 0])
        dice_liver_cumul += score_dice_liver
        iou_liver_cumul += score_iou_liver

        # Berechnung der Eingabe für das Tumor-Netzwerk
        for ich in range(opt.input_nc):
            batch_x[0, ich] = batch_x[0, ich] * model_liver.fake_B2[0, 1]
            batch_x[0, ich] -= torch.min(batch_x[0, ich])
            batch_x[0, ich] /= torch.max(batch_x[0, ich])

        # Auswertung des Tumor-Netzwerks
        true_tumor = batch_y.clone()
        true_tumor = (true_tumor == 2).float()
        data = {'A': batch_x, 'A_paths': path, 'B': true_tumor, 'B_paths': path}
        model_tumor.set_input(data)
        model_tumor.test()
        # Falls alle Pixel dem Hintergrund zugeordnet werden #
        # model_tumor.fake_B2[0, 0] = (model_tumor.fake_B2[0, 0] > 0.6).float()
        # model_tumor.fake_B2[0, 1] = (model_tumor.fake_B2[0, 1] > 0.4).float()
        ###################
        model_tumor.fake_B2[0, 0] = (model_tumor.fake_B2[0, 0] > 0.5).float()
        model_tumor.fake_B2[0, 1] = (model_tumor.fake_B2[0, 1] > 0.5).float()

        # Berechnung der Bewertungen
        score_dice_tumor = dice(model_tumor.fake_B2[0], true_tumor[0, 0])
        score_iou_tumor = iou(model_tumor.fake_B2[0], true_tumor[0, 0])
        dice_tumor_cumul += score_dice_tumor
        iou_tumor_cumul += score_iou_tumor

        # Visualisierung über visdom
        if step % opt.display_step == 0:
            visuals_liver = model_liver.get_current_visuals()
            visuals_tumor = model_tumor.get_current_visuals()
            # print("Step %d | %s" % (step, path))
            print("Step %d | Dice_liver: %f, IoU_liver: %f" % (step, score_dice_liver, score_iou_liver))
            print("Step %d | Dice_tumor: %f, IoU_tumor: %f" % (step, score_dice_tumor, score_iou_tumor))
            visualizer.display_current_results(getBothVisuals(visuals_liver, visuals_tumor), 1, False)

    # Berechnung der Bewertungsdurchschnitte
    dice_liver_complete = dice_liver_cumul / dataset_size
    iou_liver_complete = iou_liver_cumul / dataset_size
    dice_tumor_complete = dice_tumor_cumul / dataset_size
    iou_tumor_complete = iou_tumor_cumul / dataset_size
    print("--------------------------------------------------------")
    print('End score:')
    print("LIVER | Dice: %f, IoU: %f" % (dice_liver_complete, iou_liver_complete))
    print("TUMOR | Dice: %f, IoU: %f" % (dice_tumor_complete, iou_tumor_complete))
