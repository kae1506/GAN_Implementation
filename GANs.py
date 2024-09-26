"""
Implementation of GAN by Me.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torchvision
import os 

os.system('rm -rf runs')

class Discriminator(nn.Module):
    def __init__(self, lr, input_shape):
        super().__init__()
        self.lr = lr
        self.input_shape = input_shape

        self.fc_layers = nn.Sequential(
            nn.Linear(input_shape, 256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

        self.optimizer = optim.Adam(self.parameters(), lr=lr)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, x):
        t = self.fc_layers(x)

        return t

class Generator(nn.Module):
    def __init__(self, lr, input_shape, output_shape):
        super().__init__()
        self.input_shape = input_shape

        self.conv_layers = torch.nn.Sequential(
            nn.Linear(input_shape, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, output_shape),
            nn.Tanh()
        )

        self.optimizer = optim.Adam(self.parameters(), lr=lr)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, x):
        t = self.conv_layers(x)

        return t

#### Hyper Parameters ####
noise_shape = 64
mnist_shape = 28*28
lr = 3e-4
batch_size = 32
epochs = 15

fixed_noise = torch.randn(batch_size, noise_shape).to('cuda')

generator = Generator(lr, noise_shape, mnist_shape)
discriminator = Discriminator(lr, mnist_shape)

transform = transforms.Compose([
    transforms.ToTensor(), transforms.Normalize((0.5, ), (0.5, )),
])

dataset = datasets.MNIST(root="dataset/", transform=transform, download=True)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

log_loss = nn.BCELoss()

fake_writer = SummaryWriter(f"runs/GAN_MNIST_CNN/fake")
real_writer = SummaryWriter(f"runs/GAN_MNIST_CNN/real")
step = 0



for epoch in range(epochs):
    loop = tqdm(enumerate(loader), total=len(loader))
    for batch_idx, (real, _) in loop:
        ''' Updating 
        1. The discriminator's loss
            maximize log(d(z)) + log(1-d(g(x)))
        
        2. The Generator's Loss
            maximize = log(d(g(x))
        
        Note: We are using binary cross entropy with ones_like for log
        '''
        #print(real.shape)

        discriminator.optimizer.zero_grad()
        generator.optimizer.zero_grad()

        noise = torch.randn(batch_size, noise_shape).to(generator.device)
        real = real.to(discriminator.device).reshape(-1, mnist_shape)
        
        fake = generator.forward(noise).reshape(batch_size, mnist_shape)

        fake_pred = discriminator.forward(fake.detach()).view(-1)
        real_pred = discriminator.forward(real).view(-1)

        loss1 = log_loss(real_pred, torch.ones_like(real_pred)) # calculating log(real_pred)
        loss2 = log_loss(fake_pred, torch.zeros_like(fake_pred))
        lossD = (loss1 + loss2) / 2
        
        lossD.backward(retain_graph=True)
        discriminator.optimizer.step()

        output = discriminator(fake).view(-1)
        lossG = log_loss(output, torch.ones_like(output))
        
        print(lossG, lossD)
        
        lossG.backward()
        generator.optimizer.step()

        loop.set_description(f'Epoch {epoch}/{epochs}')
        loop.set_postfix(lossG=lossG.item(), lossD=lossD.item())
        
        if batch_idx == 0:
            print(f'Epoch {epoch}/{epochs}, lossG={lossG.item()}, lossD={lossD.item()}')

            with torch.no_grad():
                fake = generator(fixed_noise).reshape(-1, 1, 28, 28)
                data = real.reshape(-1, 1, 28, 28)
                img_grid_fake = torchvision.utils.make_grid(fake, normalize=True)
                img_grid_real = torchvision.utils.make_grid(data, normalize=True)

                fake_writer.add_image(
                        "Mnist Fake Images", img_grid_fake, global_step=step
                )
                real_writer.add_image(
                    "Mnist Real Images", img_grid_real, global_step=step
                )

                step += 1
