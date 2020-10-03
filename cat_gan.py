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

from helpers.gan import get_optimizer, run_a_gan, initialize_weights
from helpers.gan import build_dc_classifier, build_dc_generator

# WandB – Import the wandb library
import wandb

wandb.init(project="cat-gan")

# WandB – Config is a variable that holds and saves hyperparameters and inputs
config = wandb.config          # Initialize config
config.batch_size = 128
config.test_batch_size = 10    # input batch size for testing (default: 1000)
config.NUM_TRAIN = 12800
config.epochs = 215             # number of epochs to train (default: 10)
config.NUM_VAL = 3000

config.NOISE_DIM = 100
config.batch_size = 128

dtype = torch.cuda.FloatTensor
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor

def show_images(images, numItr="test", out="./out/"):
    images = np.reshape(images, [images.shape[0], -1])  # images reshape to (batch_size, D)
    unorm = UnNormalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    
    for i, img in enumerate(images):
        img = img.reshape(3, 64, 64)
        img = unorm(img).numpy()
        img = np.moveaxis(img, 0, -1)
        img = ((img - img.min()) * 255) / (img.max() - img.min())
        plt.imsave("output/" + str(numItr) + "_" + str(i) + ".png", img.astype(np.uint8))
    return 

if __name__ == "__main__":

    cat_train = dset.ImageFolder('./cats', transform=T.Compose(
                                 [ T.Resize(64),
                                  T.CenterCrop(64),
                                  T.ToTensor(),
                                  T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                 ]
                                ))

    loader_train = DataLoader(cat_train, batch_size=config.batch_size,
                              shuffle=True, pin_memory=True, num_workers=2)

    imgs = loader_train.__iter__().next()[0].view(config.batch_size, 3, 4096).squeeze()
    show_images(imgs)
    
    print("Starting training now")
    
    images, G_losses, D_losses = run_a_gan(loader_train, batch_size=config.batch_size, show_every=10000, noise_size=config.NOISE_DIM, num_epochs=4000)


    plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses,label="G")
    plt.plot(D_losses,label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("output/loss.png")
