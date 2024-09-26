"""
GANS WITH CNNS, tHE RIGHT WAY.
"""

import torch
import torch.nn as nn
from matplotlib.pyplot import imshow, plot, show
import torch.optim as optim
from tqdm import tqdm
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torchvision

def logged(inputs, reduction='mean'):
    output = torch.log(inputs)
    for i in range(len(outputs)):
        if output[i] > 100.0000:
            output[i] = 100.0000
        elif output[i] < -100.0000:
            output[i] = -100.0000
    if reduction:
        return -output.mean()
    else:
        return -output

class Discriminator(nn.Module):
    def __init__(self, input_shape, features_d):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape, features_d, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            self.make_block(features_d, features_d*2, 4, 2, 1),
            self.make_block(features_d*2, features_d*4, 4, 2, 1),
            self.make_block(features_d*4, features_d*8, 4, 2, 1),
            nn.Conv2d(features_d*8, 1, kernel_size=4, stride=2, padding=0),
            nn.Sigmoid()
        )

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
                nn.init.normal_(m.weight.data, 0.0, 0.2)

    def make_block(self, in_c, out_c, k, s, p):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=k, stride=s, padding=p),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        return self.conv(x)

#disc = Discriminator(1, 16)
#inps = torch.randn(5, 1, 64, 64)
#outs = disc.forward(inps)
#print(outs.shape)
#quit()

class Generator(nn.Module):
    def __init__(self, input_shape, output_shape, features_d):
        super().__init__()

        self.conv = nn.Sequential(
            self.make_block(input_shape, features_d*16, 4, 1, 0),
            self.make_block(features_d*16, features_d*8, 4, 2, 1),
            self.make_block(features_d*8, features_d*4, 4, 2, 1),
            self.make_block(features_d*4, features_d*2, 4, 2, 1),
            nn.ConvTranspose2d(
                features_d*2,
                output_shape,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.Tanh()
        )

        self.convt1 = nn.ConvTranspose2d(input_shape, features_d*16, 4, 1, 0)
        self.convt2 = nn.ConvTranspose2d(features_d*16, features_d*8, 4, 2, 1)
        self.convt3 = nn.ConvTranspose2d(features_d*8, features_d*4, 4, 2, 1)
        self.convt4 = nn.ConvTranspose2d(features_d*4, features_d*2, 4, 2, 1)
        self.convt5 = nn.ConvTranspose2d(features_d*2, 1, 4, 2, 1)


        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
                nn.init.normal_(m.weight.data, 0.0, 0.2)


    def make_block(self, in_c, out_c, k, s, p):
        return nn.Sequential(
            nn.ConvTranspose2d(in_c, out_c, kernel_size=k, stride=s, padding=p),
            nn.BatchNorm2d(out_c),
            nn.ReLU()
        )

    def forward(self, x):
        x_ = x.clone()
        #x = self.convt1(x)
        #print(x.shape)
        #x = self.convt2(x)
        #print(x.shape)
        #x = self.convt3(x)
        #print(x.shape)
        #x = self.convt4(x)
        #print(x.shape)
        #x = self.convt5(x)
        #print(x.shape)


        return self.conv(x_)
#gen = Generator(100, 1, 64)
#inps = torch.randn(1, 100, 1, 1)
#outs = gen.forward(inps)
#print(outs.shape)
#quit() 

# Hyper Parameters
batch_size = 32
channels = 1
noise_shape = (batch_size, 100, 1, 1)
H, W = 64, 64
lr = 2e-4
epochs = 15

# Models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

disc = Discriminator(channels, 64).to(device)
gen = Generator(noise_shape[1], channels, 64).to(device)

loss = nn.BCELoss()
disc_optim = optim.Adam(disc.parameters(), lr=lr, betas=(0.5, 0.999))
gen_optim = optim.Adam(gen.parameters(), lr=lr, betas=(0.5, 0.999))

# Dataset
transforms = transforms.Compose([
    transforms.Resize(H),
    transforms.ToTensor(),
    transforms.Normalize([0.5 for _ in range(channels)], [0.5 for _ in range(channels)])
])
dataset = datasets.MNIST(root="dataset/", transform=transforms, download=True)
test = datasets.MNIST(root="dataset/", transform=transforms, train=False, download=True)

loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test, batch_size=batch_size, shuffle=True)

fixed_noise = torch.randn(noise_shape).to('cuda')

fake_writer = SummaryWriter(f"runs/DCGAN_MNIST/fake", flush_secs=1)
real_writer = SummaryWriter(f"runs/DCGAN_MNIST/real", flush_secs=1)
step = 0

disc.train()
gen.train()

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
        # print(real.shape)

        disc_optim.zero_grad()
        gen_optim.zero_grad()

        noise = torch.randn(noise_shape).to(device)
        real = real.to(device)
        fake = gen.forward(noise).reshape(batch_size, 1, H, W)

        fake_pred = disc.forward(fake.detach()).view(-1)
        real_pred = disc.forward(real).view(-1)

        #print(fake_pred, real_pred)

        loss1 = loss(real_pred, torch.ones_like(real_pred))  # calculating log(real_pred)
        loss2 = loss(fake_pred, torch.zeros_like(fake_pred))
        lossD = (loss1 + loss2) / 2
        
        #print(lossD)
        
        lossD.backward(retain_graph=True)
        disc_optim.step()

        output = disc(fake).view(-1)
        lossG = loss(output, torch.ones_like(output))
        #print(lossG)

        lossG.backward()
        gen_optim.step()

        loop.set_description(f'Epoch {epoch}/{epochs}')
        loop.set_postfix(lossG=lossG.item(), lossD=lossD.item())

        with torch.no_grad():
            fake = gen(fixed_noise)
            img_grid_fake = torchvision.utils.make_grid(fake, normalize=True)
            img_grid_real = torchvision.utils.make_grid(real, normalize=True)

            fake_writer.add_image(
                "Mnist Fake Images", img_grid_fake, global_step=step
            )
            real_writer.add_image(
                "Mnist Real Images", img_grid_real, global_step=step
            )

            step += 1

# input = torch.randn((1, 3, 64, 64)).to(device)
# print(disc.forward(input).reshape(-1))
# input = torch.randn((1,100,1,1)).to(device)
# print(gen.forward(input).shape)
# imshow(gen.forward(input).reshape((64, 64, 3)).detach().cpu().numpy())
# show()
# imshow(input.reshape(1, 100).detach().cpu().numpy())
# show()
