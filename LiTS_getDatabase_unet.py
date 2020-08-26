from __future__ import print_function, division, absolute_import, unicode_literals
import scipy.io as sio
import numpy as np
import os, os.path
from image_util_unet import BaseDataProvider

"""
Aus der originalen Implementation
Enthält die Klasse DataProvider_LiTS
"""


class DataProvider_LiTS(BaseDataProvider):
    """
    Erbt von BaseDataProvider
    Durchsucht das über path gegebene Verzeichnis nach Bildern
    Gibt diese als numpy-Array zurück und führt Data Augmentation durch
    """

    n_class = 2

    def __init__(self, inputSize,fineSize, segtype, semi_rate, input_nc, path, a_min=0, a_max=100, mode=None):
        super(DataProvider_LiTS, self).__init__(a_min, a_max)
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
        """
        Durchsucht den Pfad path_ nach Bildern und erzeugt eine Liste mit Pfaden
        und Dateinamen der Bilder
        """

        path_ = os.path.join(self.path, self.mode)
        filefolds = os.listdir(path_)
        self.imageNum = []
        self.filePath = []

        for isub, filefold in enumerate(filefolds):
            # if isub % 10 != 0: continue  # Für Weglassen von Daten bei Training ohne msloss

            foldpath = os.path.join(path_, filefold)
            dataFold = sorted(os.listdir(foldpath))
            for inum, idata in enumerate(dataFold):
                dataNum = int(idata.split('.')[0])
                dataFold[inum] = dataNum
            dataFile = sorted(dataFold)
            for islice in range(1, len(dataFile)-1):
                filePath = os.path.join(foldpath, str(dataFile[islice]) + '.mat')
                # Laden des Bildes und des Labels
                file = sio.loadmat(filePath)

                data = file['data']
                label = file['labels']
                # Prüft, ob mindestens ein Pixel in Bild und Label nicht 0 ist
                if np.amax(data) == 0: continue
                if np.amax(label) == 0: continue
                if self.segtype == "tumor": # or self.segtype == "both":
                    if np.amax(label)!=2: continue  # AUSKOMMENTIEREN FÜR ALLE BILDER BEI TUMOR
                self.imageNum.append((foldpath, dataFile[islice], isub))

        if self.mode == "train":
            # Im Training wird die Liste der Bilder zufällig permutiert
            np.random.shuffle(self.imageNum)

        return len(self.imageNum)

    def _shuffle_data_index(self):
        """
        Erhöhung des Attributs self.data_idx (aktueller Stand in der Liste der Bilder)
        Ist die Liste einmal durchgegangen, wird die Liste im Training zufällig permutiert
        """
        self.data_idx += 1
        if self.data_idx >= self.n_data:
            self.data_idx = 0
            if self.mode =="train":
                np.random.shuffle(self.imageNum)

    def _next_data(self):
        """
        Gibt das Bild und Label als numpy-Array zurück, die an der Stelle self.data_idx
        in der Liste stehen
        Ist für das Weglassen von Labeln zuständig, ebenso zur Erzeugung des korrekten
        Labels für die aktuelle Segmentierungsaufgabe
        """
        self._shuffle_data_index()
        filePath = self.imageNum[self.data_idx]
        data = np.zeros((self.nx, self.ny, self.channels))
        labels = np.zeros((self.nx, self.ny, self.channels))

        for ich in range(self.channels):
            fileName = os.path.join(filePath[0], str(filePath[1]-1+ich) + '.mat')
            # Laden des Bildes und des Labels
            file = sio.loadmat(fileName)

            data[:, :, ich] = file['data']
            labels[:, :, ich] = file['labels']

        # Begrenzung der Pixelwerte auf das Intervall [-124, 276]
        data = np.clip(data+124, 0, 400)

        # EINGEFÜGT, UM LABEL KORREKT WEGZULASSEN, VORHER IN _augment_data
        if self.segtype == "liver":
            # Wird das Netzwerk trainiert/getestet, das in die Klassen "Hintergrund"
            # und "Leber/Tumor" segmentiert, werden die Klassen "Leber/Tumor" im
            # Label zusammengefasst
            labels = (labels[...,1]>0).astype(float)

        elif self.segtype == "tumor":
            # Wird das Netzwerk trainiert, das in die Klassen "Leber" und "Tumor"
            # segmentiert, wird als Eingabebild ein Bild verwendet, bei dem alle
            # Hintergrundpixel den Wert 0 haben.
            for ich in range(self.channels):
                data[..., ich] = data[..., ich] * (labels[..., ich] > 0)  # AUSKOMMENTIEREN FÜR SINGLE-NETWORK-TRAINING
            labels = (labels[...,1] == 2).astype(float)

        elif self.segtype == 'both':
            # Wird eine Kombination aus zwei Netzwerken getestet, wird das Label
            # der mittleren Eingabeschicht beibehalten
            labels = labels[..., 1]

        # WEGLASSEN DES LABELS
        if filePath[-1] % self.semi_rate != 0:
            labels = np.zeros_like(labels)
        # EINGEFÜGTER TEIL ENDE

        path = filePath[0] + str(filePath[1])
        return data, labels, path

    def _augment_data(self, data, labels):
        """
        Durchführung von Data Augmentation durch Rotationen und Spiegelungen
        Downsampling der Eingabe
        """

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

            # Rotation von Eingabe und Label um ein zufälliges Vielfaches von 90°
            op = np.random.randint(0, 4)  # 0, 1, 2, 3
            data, labels = np.rot90(data, op), np.rot90(labels, op)

            # Flip horizontal / vertikal
            op = np.random.randint(0, 3)  # 0, 1
            if op < 2:
                data, labels = np.flip(data, op), np.flip(labels, op)

        return data, labels
