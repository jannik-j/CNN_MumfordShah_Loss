import torch


"""
Selbst verfasst
Funktionen zur Berechnung der IoU- und Dice-Bewertung der Netzwerke
"""


def iou(out, true, smooth=1):
    """ IoU-Bewertung """
    # out: Ausgabe des Netzwerks, Dimensionen k x b x h
    # true: Label, Dimensionen b x h. Die Einträge stehen für die korrekte Klasse
    # b: Breite, h: Höhe, k: Anzahl der Klassen
    # smooth: Additionsparameter, damit für union=0 nicht durch 0 geteilt wird
    true_ = torch.zeros_like(out)
    for c in range(out.shape[0]):
        true_[c] = (true == c).float()
    out_ = out.clone().detach()
    intersection = torch.sum(torch.abs(out_*true_), (1, 2))
    union = torch.sum(out_, (1, 2)) + torch.sum(true_, (1, 2)) - intersection
    iou = torch.mean((intersection+smooth) / (union+smooth), 0)
    return iou.item()


def dice(out, true, smooth=1):
    """ Dice-Bewertung """
    # out: Ausgabe des Netzwerks, Dimensionen k x b x h
    # true: Label, Dimensionen b x h. Die Einträge stehen für die korrekte Klasse
    # b: Breite, h: Höhe, k: Anzahl der Klassen
    # smooth: Additionsparameter, damit für union=0 nicht durch 0 geteilt wird
    true_ = torch.zeros_like(out)
    for c in range(out.shape[0]):
        true_[c] = (true == c).float()
    out_ = out.clone().detach()
    intersection = torch.sum(torch.abs(out_*true_), (1, 2))
    union = torch.sum(out_, (1, 2)) + torch.sum(true_, (1, 2))
    dice = torch.mean((2.*intersection + smooth) / (union+smooth), 0)
    return dice.item()
