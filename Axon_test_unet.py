from options.train_options import TrainOptions
from Axon_getDatabase_unet import DataProvider_Axon
from models.models import create_model
from util.visualizer import Visualizer
from collections import OrderedDict
import torch
from scores import dice, iou
import util.util as util

"""
Selbst verfasst, in Anlehnung an LiTS_test_unet.py
Skript zum Testen auf dem Datensatz zur Axon-Myelin-Segmentierung
"""


def getVisuals(model):
    """
    Hilsfunktion, um die aktuellen Ein- und Ausgaben in das Netzwerk über visdom
    korrekt darzustellen
    """
    input = util.tensor2im(model.input_A[:, 0])

    label = util.tensor2im(model.input_B[:, 0]) / 2

    background = util.tensor2im(model.fake_B2[:, 0])
    myelin = util.tensor2im(model.fake_B2[:, 1])
    axon = util.tensor2im(model.fake_B2[:, 2])

    visuals = OrderedDict([('input', input), ('label', label), ('background', background),
                           ('myelin', myelin), ('axon', axon)])
    return visuals


# Parsen der Kommandozeilenargumente
opt = TrainOptions().parse()
opt.isTrain = False
input_min       = 0
input_max       = 400

# Laden des gespeicherten Netzwerks
model = create_model(opt)
# Initialisierung eines visdom-Session
visualizer = Visualizer(opt)

# Daten über die Klasse DataProvider_Axon laden
data_train = DataProvider_Axon(opt.inputSize, opt.fineSize, opt.segType, opt.semi_rate, opt.input_nc,
                               opt.dataroot, a_min=input_min, a_max=input_max, mode="test")
dataset_size = data_train.n_data
print('#testing images = %d' % dataset_size)
# Variablen, um den Gesamtdurchschnitt der Bewertungen zu berechnen
dice_cumul, iou_cumul = .0, .0

for step in range(dataset_size):
    # Auswertung auf den Daten im test set und Berechnung der Bewertungen

    batch_x, batch_y, path = data_train(opt.batchSize)
    data = {'A': batch_x, 'A_paths': path, 'B': batch_y, 'B_paths': path}
    model.set_input(data)
    model.test()

    # Berechnung des Maximums des Softmax Layers für jeden Pixel, um die vom
    # Netzwerk berechnete Segmentierung zu erhalten
    out = torch.argmax(model.fake_B2[0], dim=0)
    background = (out == 0).float()
    myelin = (out == 1).float()
    axon = (out == 2).float()
    map = torch.stack((background, myelin, axon))

    # Berechnung der Bewertungen
    score_dice = dice(map, batch_y[0, 0])
    score_iou = iou(map, batch_y[0, 0])
    dice_cumul += score_dice
    iou_cumul += score_iou

    # Visualisierung über visdom
    if step % opt.display_step == 0:
        print("Step %d | Dice: %f, IoU: %f" % (step, score_dice, score_iou))
        visualizer.display_current_results(getVisuals(model), 1, False)

# Berechnung der Bewertungsdurchschnitte
dice_complete = dice_cumul / dataset_size
iou_complete = iou_cumul / dataset_size
print("--------------------------------------------------------")
print('End score:')
print("Dice: %f, IoU: %f" % (dice_complete, iou_complete))
