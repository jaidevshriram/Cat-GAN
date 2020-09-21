import torch
import torch.nn as nn
from torch.nn import init
import torchvision
import torchvision.transforms as T
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler
import torchvision.datasets as dset
import CatDataset as cdset

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# %matplotlib inline
# plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
# plt.rcParams['image.interpolation'] = 'nearest'
# plt.rcParams['image.cmap'] = 'gray'

# %load_ext autoreload
# %autoreload 2

def show_images(images, numItr, out="./out/"):
    images = np.reshape(images, [images.shape[0], -1])  # images reshape to (batch_size, D)
    sqrtn = int(np.ceil(np.sqrt(images.shape[0])))
    sqrtimg = int(np.ceil(np.sqrt(images.shape[1])))

    fig = plt.figure(figsize=(sqrtn, sqrtn))
    gs = gridspec.GridSpec(sqrtn, sqrtn)
    gs.update(wspace=0.05, hspace=0.05)

    for i, img in enumerate(images):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        img = img.reshape(3, 64, 64)
        img = np.moveaxis(img, 0, -1)
        plt.imsave("output/" + str(numItr) + "_" + str(i) + ".png", img)
    return 

NUM_TRAIN = 12800
NUM_VAL = 3000

NOISE_DIM = 96
batch_size = 128

cat_train = cdset.CatDataset('./cats')
loader_train = DataLoader(cat_train, batch_size=batch_size,
                          shuffle=True)

cat_val = cdset.CatDataset('./cats')
loader_val = DataLoader(cat_val, batch_size=batch_size,
                        shuffle=True)

# imgs = loader_train.__iter__().next().view(batch_size, 3, 4096).numpy().squeeze()
# show_images(imgs)

from helpers.gan import Flatten, Unflatten, initialize_weights
from helpers.gan import discriminator_loss, generator_loss
from helpers.gan import get_optimizer, run_a_gan

dtype = torch.cuda.FloatTensor ## UNCOMMENT THIS LINE IF YOU'RE ON A GPU!

from helpers.gan import build_dc_classifier
from helpers.gan import build_dc_generator

D_DC = build_dc_classifier(batch_size).type(dtype) 
D_DC.apply(initialize_weights)
G_DC = build_dc_generator(NOISE_DIM).type(dtype)
G_DC.apply(initialize_weights)

D_DC_solver = get_optimizer(D_DC, 0.0005)
G_DC_solver = get_optimizer(G_DC, 0.0002)

images, G_losses, D_losses = run_a_gan(D_DC, G_DC, D_DC_solver, G_DC_solver, discriminator_loss, generator_loss, loader_train, noise_size=NOISE_DIM, num_epochs=100)

torch.save(D_DC_solver, "./d_trained.pth")
torch.save(G_DC_solver, "./g_trained.pth")

plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses,label="G")
plt.plot(D_losses,label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.savefig("output/loss.png")


numIter = 0
for img in images:
    print("Iter: {}".format(numIter))
    show_images(img, numIter)
    numIter += 250
    print()
    
show_images(images[-1], "final")
