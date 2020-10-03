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
ngpu = 1

# dtype = torch.FloatTensor
dtype = torch.cuda.FloatTensor
device = torch.device("cuda:0")

# https://discuss.pytorch.org/t/simple-way-to-inverse-transform-normalization/4821
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
    tensor = torch.randn(batch_size, dim, 1, 1, device = device)
    return tensor

def get_optimizer(model, lr=0.02):
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.5, 0.999))
    return optimizer

def build_dc_classifier():

    model = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
        )
    
    return model

def build_dc_generator(noise_dim=NOISE_DIM):
    
    model = nn.Sequential(
            nn.ConvTranspose2d(noise_dim, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh(),
        )
        
    return model

def bce_loss(output, target):
    return nn.BCELoss()(output, target)

def discriminator_loss(logits_real, logits_fake):
    
    fake_labels = torch.zeros(logits_fake.shape).type(dtype) * (1-0.1)
    loss = bce_loss(logits_fake, fake_labels)
    
    wandb.log({"D_fake_loss": loss.item()})

    true_labels = torch.ones(logits_real.shape).type(dtype) * (1-0.1)
    loss += bce_loss(logits_real, true_labels)
    
    wandb.log({"D_real_loss": bce_loss(logits_real, true_labels).item()})
    
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

def initialize_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def show_grid(img, i):
    unorm = UnNormalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    img = unorm(img)
    npimg = img.detach().numpy()
    npimg *= 255
    plt.imsave("grid/" + str(i) + ".png", np.transpose(npimg, (1,2,0)).astype(np.uint8))

def run_a_gan(loader_train, show_every=1000, batch_size=250, noise_size=96, num_epochs=100):
    """
    Train a GAN!
    
    - show_every: Show samples after every show_every iterations.
    - batch_size: Batch size to use for training.
    - noise_size: Dimension of the noise to use as input to the generator.
    - num_epochs: Number of epochs over the training dataset to use for training.
    """
    
    D = build_dc_classifier().type(dtype) 
    D.apply(initialize_weights)
    G = build_dc_generator(noise_size).type(dtype)
    G.apply(initialize_weights)

    # Handle multi-gpu if desired
    if (device.type == 'cuda') and (ngpu > 1):
        G_DC = nn.DataParallel(G_DC, list(range(ngpu)))
        D_DC = nn.DataParallel(D_DC, list(range(ngpu)))

    D_solver = get_optimizer(D, 0.00005)
    G_solver = get_optimizer(G, 0.0002)
       
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
                        "G_optimizer_state_dict": G_solver.state_dict(),
                        "G_loss" : G_losses,
                        "D_loss": D_losses,
                        },"./model.h5")
                    
        for i, (x, y) in enumerate(loader_train, 0):

            D.train()
            G.train()
            
            # Train discriminator with real images
            D_solver.zero_grad()
            real_data = x.to(device).type(dtype)
            output_real = D(real_data)
                       
            # Train discriminator with fake images
            g_fake_seed = sample_noise(len(x), noise_size).type(dtype)
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