# import torch
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

"""
def iou(out, true, smooth=1):
    true_ = torch.stack((true.clone().detach(), true.clone().detach()))
    true_[0] = 1 - true_[0]
    intersection = torch.sum(torch.abs(out*true_), (1, 2))
    union = torch.sum(out, (1, 2)) + torch.sum(true_, (1, 2)) - intersection
    iou = torch.mean((intersection+smooth) / (union+smooth), 0)
    return iou.cpu().numpy()


def dice(out, true, smooth=1):
    true_ = torch.stack((true.clone().detach(), true.clone().detach()))
    true_[0] = 1 - true_[0]
    intersection = torch.sum(torch.abs(out*true_), (1, 2))
    union = torch.sum(out, (1, 2)) + torch.sum(true_, (1, 2))
    dice = torch.mean((2.*intersection + smooth) / (union+smooth), 0)
    return dice.cpu().numpy()
"""

"""
# Grafik Heaviside-Funktion
def H(x):
    return (x >= 0).astype(float)


def Heps(x, epsilon):
    return 1/2 * (1 + 2/np.pi * np.arctan(x/epsilon))


x = np.linspace(-2, 2, 1000)
plt.plot(x, H(x))
plt.plot(x, Heps(x, .1))
plt.plot(x, Heps(x, .01))
plt.legend(['H(x)', r'$H_{0.1}(x)$', r'$H_{0.01}(x)$'])
plt.xlabel('x')
plt.show()
"""
"""
Ausgeben aller Slices, die ein Tumor-Label haben
for i in range(1, 606):
    filename = "F:/Daten_Bachelorarbeit/train/13/"+str(i)+'.mat'
    file = sio.loadmat(filename)
    data = file['data']
    data = np.clip(data+124, 0, 400)
    label = file['labels']
    ax1 = plt.subplot(121)
    ax2 = plt.subplot(122)
    if np.amax(label) == 2:
        print(i)
        ax1.imshow(data, cmap='gray')
        ax2.imshow(label)
        plt.axis('off')
        plt.show()
"""
"""
Beispiel für Schicht und Label
file = sio.loadmat('F:/Daten_Bachelorarbeit/train/13/363.mat')
data = file['data']
data = np.clip(data+124, 0, 400)
label = file['labels']
ax1 = plt.subplot(121)
ax2 = plt.subplot(122)
ax1.imshow(data, cmap='gray')
ax1.axis('off')
ax2.imshow(label)
ax2.axis('off')
plt.show()
"""
"""
file = sio.loadmat('F:/Daten_Bachelorarbeit/train/13/363.mat')
data = file['data']
data = np.clip(data+124, 0, 400)
label = file['labels']
ax1 = plt.subplot(131)
ax2 = plt.subplot(132)
ax3 = plt.subplot(133)
ax1.imshow(data, cmap='gray')
ax2.imshow(label>0, cmap='gray')
ax3.imshow(label==2, cmap='gray')
ax1.axis('off')
ax2.axis('off')
ax3.axis('off')
plt.show()
"""

x = np.linspace(-5, 5, 1000)
plt.plot(x, np.clip(x, 0, None))
plt.xlabel(r'$z$')
plt.ylabel(r'$r(z)$')
plt.axis('equal')
plt.show()

"""
file = sio.loadmat('F:/Daten_Bachelorarbeit/example_tumor/118/243.mat')
label = file['labels']
plt.imshow(label, cmap='gray')
plt.axis('off')
plt.show()
"""
