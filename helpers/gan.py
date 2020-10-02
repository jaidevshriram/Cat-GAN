import numpy as np

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
import torch.optim as optim
from torch.utils.data import sampler
import torchvision.utils as vutils

import PIL

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import wandb

from .grad_flow import plot_grad_flow

NOISE_DIM = 96

# dtype = torch.FloatTensor
dtype = torch.cuda.FloatTensor ## UNCOMMENT THIS LINE IF YOU'RE ON A GPU!
device = torch.device("cuda:0")

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

def sample_noise(batch_size, dim, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
        
    tensor = torch.randn(batch_size, dim, 1, 1, device = device)
    return tensor

def get_optimizer(model, lr=0.0002):
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.5, 0.999))
    return optimizer

def build_dc_classifier(batch_size):

    model = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
        )
    
    return model

def build_dc_generator(noise_dim=NOISE_DIM):
    
    model = nn.Sequential(
            nn.ConvTranspose2d(noise_dim, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh(),
        )
        
    return model

def bce_loss(output, target):
    neg_abs = - output.abs()
    loss = output.clamp(min=0) - output * target + (1 + neg_abs.exp()).log()
    
    return loss.mean()

def discriminator_loss(logits_real, logits_fake):
    
    fake_labels = torch.zeros(logits_fake.shape).type(dtype) * (1-0.1)
    loss = bce_loss(logits_fake, fake_labels)

    true_labels = torch.ones(logits_real.shape).type(dtype) * (1-0.1)
    loss += bce_loss(logits_real, true_labels)
    
    return loss

def generator_loss(logits_fake):
    true_labels = torch.ones(logits_fake.shape).type(dtype)
    loss = bce_loss(logits_fake, true_labels)

    return loss

def show_images(images, numItr="test", out="./out/"):
    images = torch.reshape(images, [images.shape[0], -1])  # images reshape to (batch_size, D)
    unorm = UnNormalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    
    for i, img in enumerate(images):
        img = img.reshape(3, 64, 64)
        img = unorm(img).detach().numpy()
        img = np.moveaxis(img, 0, -1)

        img = ((img - img.min()) * 255) / (img.max() - img.min())

        plt.imsave("output/" + str(numItr) + "_" + str(i) + ".jpg", img.astype(np.uint8))

    return 

def show_grid(img, i):
    unorm = UnNormalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    img = unorm(img)
    npimg = img.detach().numpy()
    npimg *= 255
    plt.imsave("grid/" + str(i) + ".png", np.transpose(npimg, (1,2,0)).astype(np.uint8))


def run_a_gan(D, G, D_solver, G_solver, discriminator_loss, generator_loss, loader_train, show_every=1000, 
              batch_size=250, noise_size=96, num_epochs=100):
    """
    Train a GAN!
    
    Inputs:
    - D, G: PyTorch models for the discriminator and generator
    - D_solver, G_solver: torch.optim Optimizers to use for training the
      discriminator and generator.
    - discriminator_loss, generator_loss: Functions to use for computing the generator and
      discriminator loss, respectively.
    - show_every: Show samples after every show_every iterations.
    - batch_size: Batch size to use for training.
    - noise_size: Dimension of the noise to use as input to the generator.
    - num_epochs: Number of epochs over the training dataset to use for training.
    """
    
    
    D.train()
    G.train()
    
    images = []
    iter_count = 0
    
    D_losses = []
    G_losses = []
    
    wandb.watch(D, log="all")
    wandb.watch(G, log="all")
               
    for epoch in range(num_epochs):
        
        if epoch % 10 == 0:
            
            torch.save({
                        "epoch": epoch,
                        "D_state_dict": D.state_dict(),
                        "G_state_dict": G.state_dict(),
                        "D_optimizer_state_dict": D_solver.state_dict(),
                        "G_optimizer_state_dict": G_solver.state.dict(),
                        "G_loss" : G_losses,
                        "D_loss": D_losses,
                        },"./d_model.h5")
                    
        for i, (x, y) in enumerate(loader_train, 0):
            if len(x) != batch_size:
                continue
                
            # Establish convention for real and fake labels during training
            real_label = 1.
            fake_label = 0. 
            smooth  = 0.1
            
            # Train discriminator with real images
            D_solver.zero_grad()
            real_data = x.to(device).type(dtype)
            output_real = D(real_data).view(-1)
                       
            # Train discriminator with fake images
            g_fake_seed = sample_noise(batch_size, noise_size).type(dtype)
            fake_images = G(g_fake_seed).detach()
            output_fake = D(fake_images)
            
            d_error = discriminator_loss(output_real, output_fake)
            d_error.backward()
            D_solver.step()
            
            # Train Generator
            
            G_solver.zero_grad()
            g_fake_seed = sample_noise(batch_size, noise_size).type(dtype)
            fake_images = G(g_fake_seed)
            
            gen_logits_fake = D(fake_images)
            g_error = generator_loss(gen_logits_fake)
            g_error.backward()
            G_solver.step()
            
            if (iter_count % show_every == 0):
                
                fixed_noise = torch.randn(64, noise_size, 1, 1, device=device)
                
                print('Epoch [{}/{}], Iter: {}, D: {:.6}, G:{:.6}'.format(epoch, num_epochs, iter_count, d_error.item(), g_error.item()))
                show_images(fake_images[0:16].cpu(), iter_count)
                
                images = vutils.make_grid(fake_images[0:16].cpu(), padding=2, normalize=True)
                show_grid(images, iter_count)
                
#                 plot_grad_flow(G.named_parameters(), "gen_" + str(iter_count))
#                 plot_grad_flow(D.named_parameters(), "disc_" + str(iter_count))
                
            iter_count += 1
        
            wandb.log({"G_Loss": g_error.item(), "epoch": epoch})
            wandb.log({"D_Loss": d_error.item(), "epoch": epoch})
                        
            G_losses.append(g_error.item())
            D_losses.append(d_error.item())
    
    return [images], G_losses, D_losses

class Flatten(nn.Module):
    def forward(self, x):
        N, C, H, W = x.size() # read in N, C, H, W
        return x.view(N, -1)  # "flatten" the C * H * W values into a single vector per image
    
class Unflatten(nn.Module):
    """
    An Unflatten module receives an input of shape (N, C*H*W) and reshapes it
    to produce an output of shape (N, C, H, W).
    """
    def __init__(self, N=-1, C=256, H=8, W=8):
        super(Unflatten, self).__init__()
        self.N = N
        self.C = C
        self.H = H
        self.W = W
    def forward(self, x):
#         print(x.shape)
        return x.view(self.N, self.C, self.H, self.W)

def initialize_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.ConvTranspose2d):
        nn.init.normal_(m.weight.data, 1.0, 0.02)
#         nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Conv2d):
        nn.init.normal_(m.weight.data, 0.0, 0.02) 