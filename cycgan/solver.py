import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
import os
import itertools
import numpy as np
import pandas as pd
from models import Generater, Discriminator
from utils import ReplayBuffer, Lambda_LR, decode_output
from datasets import train_loder


class Solver:
    def __init__(self, args):
        # how to use GPUs
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        num_workers = max(4 * torch.cuda.device_count(), 4)
        if device == 'cuda':
            torch.backends.cudnn.benchmark = True
            print("Let's use {} GPUs!".format(torch.cuda.device_count()))

        # models
        netG_A2B = Generater(args.n_features)
        netG_B2A = Generater(args.n_features)
        netD_A = Discriminator(args.n_features)
        netD_B = Discriminator(args.n_features)
        netG_A2B.to(device)
        netG_B2A.to(device)
        netD_A.to(device)
        netD_B.to(device)

        # losses
        criterion_GAN = nn.MSELoss()
        criterion_cycle = nn.L1Loss()
        criterion_identity = nn.L1Loss()

        # optimizers
        optimizer_G = torch.optim.Adam(itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()), lr=args.lr,
                                       betas=(0.5, 0.999), weight_decay=5e-4)
        optimizer_D_A = torch.optim.Adam(netD_A.parameters(), lr=args.lr, betas=(0.5, 0.999), weight_decay=5e-4)
        optimizer_D_B = torch.optim.Adam(netD_B.parameters(), lr=args.lr, betas=(0.5, 0.999), weight_decay=5e-4)

        # LR schedulers
        lr_scheduler_G = LambdaLR(optimizer_G, lr_lambda=Lambda_LR(args.n_epochs, 0, args.decay_epoch).step)
        lr_scheduler_D_A = LambdaLR(optimizer_D_A, lr_lambda=Lambda_LR(args.n_epochs, 0, args.decay_epoch).step)
        lr_scheduler_D_B = LambdaLR(optimizer_D_B, lr_lambda=Lambda_LR(args.n_epochs, 0, args.decay_epoch).step)

        # generated image buffer
        fake_A_buffer = ReplayBuffer()
        fake_B_buffer = ReplayBuffer()

        # dataloader
        dataloader_train = train_loder(data_A_path=args.data_A_path, data_B_path=args.data_B_path,
                                       batch_size=args.batch_size, num_workers=num_workers)

        self.args = args
        self.device = device
        self.netG_A2B = netG_A2B
        self.netG_B2A = netG_B2A
        self.netD_A = netD_A
        self.netD_B = netD_B
        self.criterion_GAN = criterion_GAN
        self.criterion_cycle = criterion_cycle
        self.criterion_identity = criterion_identity
        self.optimizer_G = optimizer_G
        self.optimizer_D_A = optimizer_D_A
        self.optimizer_D_B = optimizer_D_B
        self.lr_scheduler_G = lr_scheduler_G
        self.lr_scheduler_D_A = lr_scheduler_D_A
        self.lr_scheduler_D_B = lr_scheduler_D_B
        self.fake_A_buffer = fake_A_buffer
        self.fake_B_buffer = fake_B_buffer
        self.dataloader_train = dataloader_train

        os.makedirs('output/A', exist_ok=True)
        os.makedirs('output/B', exist_ok=True)

    def train(self):
        for epoch in range(self.args.n_epochs):
            print(f'*********************Epoch:{epoch}/{self.args.n_epochs}**********************')
            epoch_loss_G, epoch_loss_G_identity, epoch_loss_G_GAN, epoch_loss_G_cycle, epoch_loss_D, epoch_loss = \
                self.train_epoch()
            print(f'Epoch: {epoch+1}/{self.args.n_epochs}, '
                  f'train_loss_G: {epoch_loss_G:.2f}, '
                  f'train_loss_G_identity: {epoch_loss_G_identity:.2f}, '
                  f'train_loss_G_GAN: {epoch_loss_G_GAN:.2f}, '
                  f'train_loss_G_cycle: {epoch_loss_G_cycle:.2f}, '
                  f'train_loss_D: {epoch_loss_D:.2f}\n')
            self.lr_scheduler_G.step(epoch)
            self.save_ckp()
        self.eval()

    def train_epoch(self):
        epoch_loss_G = 0
        epoch_loss_G_identity = 0
        epoch_loss_G_GAN = 0
        epoch_loss_G_cycle = 0
        epoch_loss_D = 0
        epoch_loss = 0
        for idx, (real_A, real_B) in enumerate(self.dataloader_train):
            target_real = torch.ones((real_A.shape[0], 1), device=self.device, dtype=torch.float)
            target_fake = torch.zeros((real_A.shape[0], 1), device=self.device, dtype=torch.float)
            real_A = real_A.to(self.device)
            real_B = real_B.to(self.device)
            self.netG_A2B.train()
            self.netG_B2A.train()
            self.netD_A.train()
            self.netD_B.train()

            ######### Generators A2B and B2A ###########
            self.optimizer_G.zero_grad()

            # Identity loss
            # G_A2B(B) should be equal B if real B is fed
            same_B = self.netG_A2B(real_B)
            loss_identity_B = self.criterion_identity(same_B, real_B)*5
            # G_B2A(A) should be equal A if real A is fed
            same_A = self.netG_B2A(real_A)
            loss_identity_A = self.criterion_identity(same_A, real_A)*5

            # GAN loss
            fake_B= self.netG_A2B(real_A)
            pred_fake = self.netD_B(fake_B)
            loss_GAN_A2B = self.criterion_GAN(pred_fake, target_real)

            fake_A = self.netG_B2A(real_B)
            pred_fake = self.netD_A(fake_A)
            loss_GAN_B2A= self.criterion_GAN(pred_fake, target_real)

            # Cycle loss
            recovered_A = self.netG_B2A(fake_B)
            loss_cycle_ABA = self.criterion_cycle(recovered_A, real_A)*10

            recovered_B = self.netG_A2B(fake_A)
            loss_cycle_BAB = self.criterion_cycle(recovered_B, real_B)*10

            # Total loss
            loss_G = loss_identity_A + loss_identity_B \
                     + loss_GAN_A2B + loss_GAN_B2A \
                     + loss_cycle_ABA + loss_cycle_BAB
            loss_G.backward()
            self.optimizer_G.step()

            ###### Discriminator A #####
            self.optimizer_D_A.zero_grad()

            # Real loss
            pred_real = self.netD_A(real_A)
            loss_D_real = self.criterion_GAN(pred_real, target_real)

            # Fake loss
            fake_A = self.fake_A_buffer.push_and_pop(fake_A)
            pred_fake = self.netD_A(fake_A.detach())
            loss_D_fake = self.criterion_GAN(pred_fake, target_fake)

            # Total loss
            loss_D_A = (loss_D_real + loss_D_fake) * 0.5
            loss_D_A.backward()
            self.optimizer_D_A.step()

            ###### Discriminator B ######
            self.optimizer_D_B.zero_grad()

            # Real loss
            pred_real = self.netD_B(real_B)
            loss_D_real = self.criterion_GAN(pred_real, target_real)

            # Fake loss
            fake_B = self.fake_B_buffer.push_and_pop(fake_B)
            pred_fake = self.netD_B(fake_B.detach())
            loss_D_fake = self.criterion_GAN(pred_fake, target_fake)

            # Total loss
            loss_D_B = (loss_D_real + loss_D_fake)*0.5
            loss_D_B.backward()
            self.optimizer_D_B.step()

            if idx % (len(self.dataloader_train) // 20) == 0 or idx == len(self.dataloader_train) - 1:
                print(f'Batch: {idx}/{len(self.dataloader_train)}, '
                      f'loss_G: {loss_G:.3f}, '
                      f'loss_G_identity: {(loss_identity_A + loss_identity_B):.3f}, '
                      f'loss_G_GAN: {(loss_GAN_A2B + loss_GAN_B2A):.3f}, '
                      f'loss_G_cycle: {(loss_cycle_ABA + loss_cycle_BAB):.3f}, '
                      f'loss_D: {(loss_D_A + loss_D_B):.3f}')

            epoch_loss_G += loss_G.item()
            epoch_loss_G_identity += (loss_identity_A + loss_identity_B).item()
            epoch_loss_G_GAN += (loss_GAN_A2B + loss_GAN_B2A).item()
            epoch_loss_G_cycle += (loss_cycle_ABA + loss_cycle_BAB).item()
            epoch_loss_D += (loss_D_A + loss_D_B).item()
            epoch_loss = epoch_loss_G + epoch_loss_D

        epoch_loss_G /= len(self.dataloader_train)
        epoch_loss_G_identity /= len(self.dataloader_train)
        epoch_loss_G_GAN /= len(self.dataloader_train)
        epoch_loss_G_cycle /= len(self.dataloader_train)
        epoch_loss_D /= len(self.dataloader_train)
        return epoch_loss_G, epoch_loss_G_identity, epoch_loss_G_GAN, epoch_loss_G_cycle, epoch_loss_D, epoch_loss

    def eval(self):
        self.load_ckp()
        self.netG_A2B.eval()
        self.netG_B2A.eval()

        generated_As = []
        generated_Bs = []
        for i, (real_A, real_B) in enumerate(self.dataloader_train):
            fake_B = self.netG_A2B(real_A).detach().to('cpu').numpy()
            fake_A = self.netG_B2A(real_B).detach().to('cpu').numpy()

            generated_B = fake_B
            generated_A = fake_A

            generated_As.append(generated_A)
            generated_Bs.append(generated_B)
        generated_As = np.concatenate(generated_As, axis=0)
        generated_Bs = np.concatenate(generated_Bs, axis=0)
        pd.DataFrame(generated_As).to_csv(f'output/generated_A.csv', index=False)
        pd.DataFrame(generated_Bs).to_csv(f'output/generated_B.csv', index=False)

    def save_ckp(self):
        torch.save(self.netG_A2B.state_dict(), 'output/netG_A2B.pth')
        torch.save(self.netG_B2A.state_dict(), 'output/netG_B2A.pth')
        torch.save(self.netD_A.state_dict(), 'output/netD_A.pth')
        torch.save(self.netD_B.state_dict(), 'output/netD_B.pth')

    def load_ckp(self):
        self.netG_A2B.load_state_dict(torch.load('output/netG_A2B.pth', map_location=self.device))
        self.netG_B2A.load_state_dict(torch.load('output/netG_B2A.pth', map_location=self.device))
        self.netD_A.load_state_dict(torch.load('output/netD_A.pth', map_location=self.device))
        self.netD_B.load_state_dict(torch.load('output/netD_B.pth', map_location=self.device))