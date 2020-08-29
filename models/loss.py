import torch
import torch.nn as nn

"""
Aus der originalen Implementation
Enthält die Klassen levelsetLoss und gradientLoss2d zur Berechnung der
Mumford-Shah-Kostenfunktion
"""


class levelsetLoss(nn.Module):
    """ Klasse zur Berechnung des Fehlerterms aus der Mumford-Shah-Kostenfunktion """
    def __init__(self):
        super(levelsetLoss, self).__init__()

    def forward(self, output, target):
        outshape = output.shape
        tarshape = target.shape
        loss = 0.0
        for ich in range(tarshape[1]):
            target_ = torch.unsqueeze(target[:,ich], 1)
            target_ = target_.expand(tarshape[0], outshape[1], tarshape[2], tarshape[3])
            # Berechnung des Mittelwerts
            pcentroid = torch.sum(target_ * output, (2,3))/torch.sum(output, (2,3))
            pcentroid = pcentroid.view(tarshape[0], outshape[1], 1, 1)
            # Berechnung von u(i,j)-c_n
            plevel = target_ - pcentroid.expand(tarshape[0], outshape[1], tarshape[2], tarshape[3])
            # Berechnung von (u(i,j)-c_n)^2 * f(i,j)
            pLoss = plevel * plevel * output
            loss += torch.sum(pLoss)
        return loss


class gradientLoss2d(nn.Module):
    """ Klasse zur Berechnung des Regularitaetsterms aus der Mumford-Shah-Kostenfunktion """
    def __init__(self, penalty='l1'):
        super(gradientLoss2d, self).__init__()
        self.penalty = penalty

    def forward(self, input):
        # Berechnung der vertikalen Differenzen
        dH = torch.abs(input[:, :, 1:, :] - input[:, :, :-1, :])
        # Berechnung der horizontalen Differenzen
        dW = torch.abs(input[:, :, :, 1:] - input[:, :, :, :-1])
        if(self.penalty == "l2"):
            # Ggf. Verwendung der 2-Norm
            dH = dH * dH
            dW = dW * dW
        loss = torch.sum(dH) + torch.sum(dW)
        return loss
