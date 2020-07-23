from __future__ import print_function, division, absolute_import, unicode_literals

#import cv2
# import glob
import numpy as np
# from PIL import Image
# import ipdb
# import matplotlib.pyplot as plt
# import torchvision.transforms as transforms
# import scipy.misc
import torch
# import matplotlib.pyplot as plt

class BaseDataProvider(object):

    def __init__(self, a_min=None, a_max=None):
        self.a_min = a_min if a_min is not None else -np.inf
        self.a_max = a_max if a_min is not None else np.inf

    def _load_data_and_label(self):
        data, label, path = self._next_data()
        data, label = self._augment_data(data, label)
        data, labels = self._process_data_labels(data, label)

        nx = data.shape[1]
        ny = data.shape[0]
        data = data.transpose(2, 0, 1)
        return path, data.reshape(1, self.channels, nx, ny), labels.reshape(1, 1, ny, nx)

    def _process_data_labels(self, data, label):
        for ich in range(self.channels):
            if np.amax(data[..., ich]) == 0 : continue
            data[..., ich] -= float(np.amin(data[..., ich]))
            data[..., ich] /= float(np.amax(data[..., ich]))
        return data, label

    def _toTorchFloatTensor(self, img):
        img = torch.from_numpy(img.copy())
        return img

    def __call__(self, n):
        path, data, labels = self._load_data_and_label()
        nx = data.shape[3]
        ny = data.shape[2]
        # X = torch.FloatTensor(n, self.channels, nx, ny).zero_()
        X = torch.empty(n, self.channels, nx, ny, device='cuda:0').zero_()
        Y = torch.empty(n, 1, nx, ny, device='cuda:0').zero_()
        P = []

        # for ich in range(data.shape[-1]):
        for ich in range(self.channels):
            # X[0, ich] = self._toTorchFloatTensor(data[0, ich])[0]
            X[0, ich] = self._toTorchFloatTensor(data[0, ich])
        # Y[0, 0] = self._toTorchFloatTensor(labels[0, 0])[0]
        Y[0, 0] = self._toTorchFloatTensor(labels[0, 0])
        P.append(path)

        for i in range(1, n):
            if self.data_idx+1 >= self.n_data:
                break
            path, data, labels = self._load_data_and_label()

            # for ich in range(data.shape[-1]):
            for ich in range(self.channels):
                # X[i, ich] = self._toTorchFloatTensor(data[0, ich])[0]
                X[i, ich] = self._toTorchFloatTensor(data[0, ich])
            # Y[i, 0] = self._toTorchFloatTensor(labels[0, 0])[0]
            Y[i, 0] = self._toTorchFloatTensor(labels[0, 0])
            P.append(path)

        return X, Y, P
