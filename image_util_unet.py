from __future__ import print_function, division, absolute_import, unicode_literals

import numpy as np
import torch

"""
Aus der originalen Implementation
Enthält die Klasse DataProvider_LiTS
"""


class BaseDataProvider(object):
    """
    Oberklasse von DataProvider_LiTS und DataProvider_Axon
    Zusammen mit diesen zum Laden und Preprocessing der Daten zuständig
    """

    def __init__(self, a_min=None, a_max=None):
        self.a_min = a_min if a_min is not None else -np.inf
        self.a_max = a_max if a_min is not None else np.inf

    def _load_data_and_label(self):
        """
        Ruft die weiteren Funktionen der Klasse auf, um das nächste Bild und
        Label zu laden, zu verarbeiten und als numpy-Array zurückzugeben
        """
        data, label, path = self._next_data()
        data, label = self._augment_data(data, label)
        data, labels = self._process_data_labels(data, label)

        nx = data.shape[1]
        ny = data.shape[0]
        data = data.transpose(2, 0, 1)
        return path, data.reshape(1, self.channels, nx, ny), labels.reshape(1, 1, ny, nx)

    def _process_data_labels(self, data, label):
        """
        Bildet die Pixelwerte des Bildes linear auf das Intervall [0, 1] ab
        """
        for ich in range(self.channels):
            if np.amax(data[..., ich]) == 0 : continue
            data[..., ich] -= float(np.amin(data[..., ich]))
            data[..., ich] /= float(np.amax(data[..., ich]))
        return data, label

    def _toTorchFloatTensor(self, img):
        """
        Hilfsfunktion, um aus einem numpy-Array einen torch-Tensor zu erstellen
        """
        img = torch.from_numpy(img.copy())
        return img

    def __call__(self, n):
        """
        Gibt Bilder und Label des nächsten Minibatches als torch-Tensor zurück
        """
        path, data, labels = self._load_data_and_label()
        nx = data.shape[3]
        ny = data.shape[2]

        # Speicherreservierung für die torch-Tensoren auf der Grafikkarte
        X = torch.empty(n, self.channels, nx, ny, device='cuda:0').zero_() # Bild
        Y = torch.empty(n, 1, nx, ny, device='cuda:0').zero_()  # Label
        P = []

        for ich in range(self.channels):
            X[0, ich] = self._toTorchFloatTensor(data[0, ich])

        Y[0, 0] = self._toTorchFloatTensor(labels[0, 0])
        P.append(path)

        for i in range(1, n):
            if self.data_idx+1 >= self.n_data:
                break
            path, data, labels = self._load_data_and_label()

            for ich in range(self.channels):
                X[i, ich] = self._toTorchFloatTensor(data[0, ich])
            
            Y[i, 0] = self._toTorchFloatTensor(labels[0, 0])
            P.append(path)

        return X, Y, P
