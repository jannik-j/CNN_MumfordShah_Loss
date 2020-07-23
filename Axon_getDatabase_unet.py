from __future__ import print_function, division, absolute_import, unicode_literals
import os
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt


class DataProvider_Axon:

    n_class = 3

    def __init__(self, inputSize,fineSize, segtype, semi_rate, input_nc, path, a_min=0, a_max=100, mode=None):
        self.nx       = inputSize
        self.ny       = inputSize
        self.nx_f = fineSize
        self.ny_f = fineSize
        self.semi_rate = semi_rate
        self.segtype = segtype
        self.channels = input_nc
        self.path     = path
        self.mode     = mode
        self.data_idx = -1
        self.n_data = self._load_data()

    def _load_data(self):
        path_ = os.path.join(self.path, self.mode, self.segtype)
        filefolds = os.listdir(path_)
        self.imageNum = []
        self.filePath = []

        for isub, filefold in enumerate(filefolds):
            foldpath = os.path.join(path_, filefold)

            fileNameData = os.path.join(foldpath, 'image.png')
            fileNameLabel = os.path.join(foldpath, 'mask.png')
            if self.mode == 'train':
                data_raw = np.array(Image.open(fileNameData))
                labels_raw = np.array(Image.open(fileNameLabel).convert('L')) // 127
            else:
                data_raw = np.array(Image.open(fileNameData))[:1024, :2048]
                labels_raw = np.array(Image.open(fileNameLabel).convert('L'))[:1024, :2048] // 127

            if isub % self.semi_rate != 0:
                labels_raw = np.zeros_like(labels_raw)

            self.imageNum.append((foldpath, data_raw, labels_raw))

        if self.mode == "train":
            np.random.shuffle(self.imageNum)

        return len(self.imageNum)

    def _shuffle_data_index(self):
        self.data_idx += 1
        if self.data_idx >= self.n_data:
            self.data_idx = 0
            if self.mode =="train":
                np.random.shuffle(self.imageNum)

    def _next_data(self):
        self._shuffle_data_index()
        filePath, data_raw, labels_raw = self.imageNum[self.data_idx]

        shape = data_raw.shape

        if self.mode == 'train':
            data = np.zeros((self.nx, self.ny, self.channels))
            labels = np.zeros((self.nx, self.ny, 1))
            data[:, :, 0], labels[:, :, 0] = self._random_crop(data_raw, labels_raw, size=(self.nx, self.ny))
        else:
            data = np.zeros((shape[0], shape[1], self.channels))
            labels = np.zeros((shape[0], shape[1], 1))
            data[:, :, 0] = data_raw
            labels[:, :, 0] = labels_raw

        # data = np.clip(data+124, 0, 400)
        path = filePath
        return data, labels, path

    def _augment_data(self, data, labels):

        if self.mode == "train":
            # downsampling x2
            op = np.random.randint(0, 4)
            if op == 0:
                data = data[::2, ::2]
                labels = labels[::2, ::2]
            elif op == 1:
                data = data[::2, 1::2]
                labels = labels[::2, 1::2]
            elif op == 2:
                data = data[1::2, ::2]
                labels = labels[1::2, ::2]
            elif op == 3:
                data = data[1::2, 1::2]
                labels = labels[1::2, 1::2]

            # Rotation 90
            op = np.random.randint(0, 4)  # 0, 1, 2, 3
            data, labels = np.rot90(data, op), np.rot90(labels, op)

            # Flip horizon / vertical
            op = np.random.randint(0, 3)  # 0, 1
            if op < 2:
                data, labels = np.flip(data, op), np.flip(labels, op)

        else:
            data = data[::2, ::2]
            labels = labels[::2, ::2]

        return data, labels

    def _load_data_and_label(self):
        data, label, path = self._next_data()
        data, label = self._augment_data(data, label)
        data, labels = self._process_data_labels(data, label)

        nx = data.shape[1]
        ny = data.shape[0]
        data = data.transpose(2, 0, 1)
        return path, data.reshape(1, self.channels, ny, nx), labels.reshape(1, 1, ny, nx)

    def _process_data_labels(self, data, label):
        for ich in range(self.channels):
            if np.amax(data[..., ich]) == 0 : continue
            data[..., ich] -= float(np.amin(data[..., ich]))
            data[..., ich] /= float(np.amax(data[..., ich]))
        return data, label

    def _random_crop(self, data, label, size=(512, 512)):
        data_crop = np.zeros(size)
        label_crop = np.zeros(size)
        w, h = data.shape[:2]
        wneu, hneu = size
        startx, starty = np.random.randint(0, [w-wneu, h-hneu])
        data_crop = data[startx:startx+size[0], starty:starty+size[1]]
        label_crop = label[startx:startx+size[0], starty:starty+size[1]]
        return data_crop, label_crop

    def _toTorchFloatTensor(self, img):
        img = torch.from_numpy(img.copy())
        return img

    def __call__(self, n):
        path, data, labels = self._load_data_and_label()
        nx = data.shape[3]
        ny = data.shape[2]

        X = torch.empty(n, self.channels, ny, nx, device='cuda:0').zero_()
        Y = torch.empty(n, 3, ny, nx, device='cuda:0').zero_()
        P = []

        X[0, 0] = self._toTorchFloatTensor(data[0, 0])
        Y[0, 0] = self._toTorchFloatTensor(labels[0, 0])
        P.append(path)

        for i in range(1, n):
            if self.data_idx+1 >= self.n_data:
                break
            path, data, labels = self._load_data_and_label()

            X[i, 0] = self._toTorchFloatTensor(data[0, 0])
            Y[i, 0] = self._toTorchFloatTensor(labels[0, 0])
            P.append(path)

        return X, Y, P
