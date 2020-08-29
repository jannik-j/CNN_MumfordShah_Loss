import torch
from collections import OrderedDict
from torch.autograd import Variable
import util.util as util
from .base_model import BaseModel
from . import networks_unet
from .loss import *

"""
Aus der originalen Implementation
Enthält die Klasse UNetModel
"""


class UNetModel(BaseModel):
    """
    Erbt von BaseModel
    Enthält weitere Funktionen für das verwendete Netzwerk, insbesondere die
    Initialisierung und das Berechnen der Kostenfunktionen
    """

    def name(self):
        return 'UNetModel'

    def initialize(self, opt):
        """ Initilaisierung des Netzwerks """
        # Initialisierung der Oberklasse
        BaseModel.initialize(self, opt)

        nb = opt.batchSize
        size = opt.fineSize
        self.input_A = self.Tensor(nb, opt.input_nc, size, size)
        self.input_B = self.Tensor(nb, int(opt.output_nc/2), size, size).long()

        # Aufruf von networks_unet.define_G zum Laden der U-Net-Architektur
        self.net = networks_unet.define_G(opt.input_nc, opt.output_nc,
                                        opt.ngf, opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)

        # Laden eines gespeicherten Netzwerks
        if not self.isTrain or opt.continue_train:
            which_epoch = opt.which_epoch
            self.load_network(self.net, 'G_A', which_epoch)

        if self.isTrain:
            self.old_lr = opt.lr
            # Definition der Kostenfunktionen
            self.criterionCE = torch.nn.CrossEntropyLoss()
            self.criterionLS = levelsetLoss()
            self.criterionTV = gradientLoss2d()

            # Initilaisierung der Optimierer
            if opt.optim == 'Adam':
                self.optimizer_ = torch.optim.Adam(self.net.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            elif opt.optim == 'SGD':
                self.optimizer_ = torch.optim.SGD(self.net.parameters(), lr=opt.lr)
            elif opt.optim == 'RMS':
                self.optimizer_ = torch.optim.RMSprop(self.net.parameters(), lr=opt.lr)
            self.optimizers = []
            self.schedulers = []
            self.optimizers.append(self.optimizer_)
            for optimizer in self.optimizers:
                self.schedulers.append(networks_unet.get_scheduler(optimizer, opt))

        # Ausgabe der Netzwerkarchitektur in der Konsole
        print('---------- Networks initialized -------------')
        networks_unet.print_network(self.net)
        print('-----------------------------------------------')

    def set_input(self, input):
        """ Übernimmt Input-Daten aus Dictionary in das Netzwerk """
        input_A = input['A']
        input_B = input['B'].long()
        self.input_A.resize_(input_A.size()).copy_(input_A)
        self.input_B.resize_(input_B.size()).copy_(input_B)
        self.image_paths = input['A_paths']

    def set_input_test(self, input):
        """ Übernimmt Input-Daten ohne Label aus Dictionary """
        input_A = input['A']
        self.input_A.resize_(input_A.size()).copy_(input_A)
        self.image_paths = input['A_paths']

    def forward(self):
        """ Erstellt eine torch-Variable aus der Eingabe """
        self.real_A = Variable(self.input_A)
        self.real_B = Variable(self.input_B)

    def test(self):
        """Wertet das Netzwerk für die Eingabe in input_A aus, speichert
        das Ergebnis in fake_B und fake_B2"""
        real_A = (self.input_A).clone().detach()
        # Auswertung des Netzwerks
        fake_B = self.net(real_A)
        fake_B2 = torch.clamp(fake_B[:, :], 1e-10, 1.0)

        self.fake_B2 = fake_B2.data
        self.fake_B = fake_B.data

    def get_image_paths(self):
        return self.image_paths

    def backward_G(self):
        """
        Auswertung des Netzwerks, Berechnung der Kostenfunktionen, Berechnung
        der Gradienten
        """
        # Auswertung des Netzwerks
        fake_B = self.net(self.real_A)
        fake_B2 = torch.clamp(fake_B[:, :], 1e-10, 1.0)

        loss_C = 0
        numch = 0
        # Berechnung der Kreuzentropie für alle Daten aus dem Minibatch, für
        # die das Label nicht weggelassen wurde
        for ibatch in range(self.real_B.shape[0]):
            if torch.max(self.real_B[ibatch, 0]) != 0:
                realB = self.real_B[ibatch, 0].unsqueeze(0)
                fakeB = fake_B2[ibatch, :].unsqueeze(0)
                loss_C += self.criterionCE(fakeB, realB) # * 100
                numch += 1.0
        if numch > 0:
            loss_C = loss_C / numch
            self.loss_C = loss_C.item()

        else:
            self.loss_C = 0
        ##################################################
        # Berechnung der Mumford-Shah-Kostenfunktion
        loss_L = self.criterionLS(fake_B2, self.real_A)
        loss_A = self.criterionTV(fake_B2) *0.001
        # Gewichtung der Kostenfunktionen
        loss_LS = (loss_L + loss_A) * self.opt.lambda_A

        loss_tot = loss_C+loss_LS
        # Berechnung der Gradienten
        loss_tot.backward()

        self.fake_B2 = fake_B2.data
        self.loss_LS = loss_LS.item()
        ##################################################
        # Zum Training von Netzwerken ohne die Mumford-Shah-Kostenfunktion
        # muss obiger Block durch den folgenden ersetzt werden:
        ##################################################
        # loss_tot = loss_C
        # loss_tot.backward()
        # self.fake_B2 = fake_B2.data
        # self.loss_LS = 0
        ##################################################

    def optimize_parameters(self):
        """ Optimierung der Parameter über die gewählte Optimiererklasse """
        self.forward()
        self.optimizer_.zero_grad()
        self.backward_G()
        self.optimizer_.step()

    def get_current_errors(self):
        """ Ausgabe der Werte der Kostenfunktionen als OrderedDict """
        ret_errors = OrderedDict([('C', self.loss_C), ('LS', self.loss_LS)])
        return ret_errors

    def get_current_visuals(self):
        """ Ausgabe der zuletzt im Netzwerk verwendeten Bilder als OrderedDict """
        real_A1 = util.tensor2im(self.input_A[:, 0])
        real_A2 = util.tensor2im(self.input_A[:, 1])
        real_A3 = util.tensor2im(self.input_A[:, 2])

        real_B2 = util.tensor2im(self.input_B[:, 0])

        fake_B0 = util.tensor2im(self.fake_B2[:, 0])
        fake_B1 = util.tensor2im(self.fake_B2[:, 1])

        ret_visuals = OrderedDict([('real_A1', real_A1),('real_A2', real_A2), ('real_A3', real_A3),
                                   ('real_B2', real_B2), ('fake_B0', fake_B0), ('fake_B1', fake_B1)])
        return ret_visuals

    def save(self, label):
        """ Speicherung des Netzwerks über save_network aus BaseModel """
        self.save_network(self.net, 'G_A', label, self.gpu_ids)
