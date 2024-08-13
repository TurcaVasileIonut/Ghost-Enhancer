import random

import cv2
import numpy as np
import torch
import torchvision
from matplotlib import pyplot as plt
from torch import nn, optim

from Data.LowToHighDataLoader import LowToHighDataLoader
from ModelSuperResolution.Discriminator import Discriminator
from ModelSuperResolution.HighToLowGenerator import HighToLowGenerator
from ModelSuperResolution.LowToHighGenerator import LowToHighGenerator


class LowToHigh:
    def plot_losses(self, epoch, generator_losses, discriminator_losses, learning_rates):
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 3, 1)
        plt.plot(range(1, epoch+1), generator_losses, label='Generator Loss')
        plt.title('Generator Loss Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')

        plt.subplot(1, 3, 2)
        plt.plot(range(1, epoch+1), discriminator_losses, label='Discriminator Loss')
        plt.title('Discriminator Loss Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')

        if learning_rates:
            plt.subplot(1, 3, 3)
            plt.plot(range(1, epoch+1), learning_rates, label='Learning Rate')
            plt.title('Learning Rate Over Epochs')
            plt.xlabel('Epoch')
            plt.ylabel('Learning Rate')

        plt.tight_layout()
        plt.show()

    def plot_result(self, dataloader, l2h_generator):
        for i, batch in enumerate(dataloader.dataloader):
            plt.subplot(1, 3, 1)
            input_img_tensor = dataloader.dataset.denormalize(batch[1][1]).permute(1, 2, 0)
            input_img = input_img_tensor.numpy()
            input_img = (input_img * 255).astype(np.uint8)  # Scale and convert to uint8
            plt.imshow(input_img)
            plt.title('Real Image')
            plt.axis('off')

            plt.subplot(1, 3, 2)
            input_img_tensor = dataloader.dataset.denormalize(batch[0][1]).permute(1, 2, 0)
            input_img = input_img_tensor.numpy()
            input_img = (input_img * 255).astype(np.uint8)  # Scale and convert to uint8
            plt.imshow(input_img)
            plt.title('Input Image')
            plt.axis('off')

            x = batch[1].cuda()
            lr_gen = l2h_generator(x)
            lr_gen_detach = lr_gen.detach().cpu()

            plt.subplot(1, 3, 3)
            lr_img_tensor = dataloader.dataset.denormalize(lr_gen_detach[0]).permute(1, 2, 0)
            lr_img = lr_img_tensor.numpy()
            lr_img = (lr_img * 255).astype(np.uint8)
            plt.imshow(lr_img)
            plt.title('High Resolution Image')
            plt.axis('off')

            plt.show()

            cv2.imwrite("../fid/dir1/input_image.png", cv2.cvtColor(input_img, cv2.COLOR_RGBA2BGR))
            cv2.imwrite("../fid/dir2/low_res_image.png",  cv2.cvtColor(lr_img, cv2.COLOR_RGBA2BGR))
            break

    def run(self):
        seed_num = 2020
        random.seed(seed_num)
        np.random.seed(seed_num)
        torch.manual_seed(seed_num)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        max_epoch = 5000
        learn_rate = 1e-4
        alpha, beta = 1, 0.05

        l2h_generator = LowToHighGenerator().cuda()
        l2h_discriminator = Discriminator(64).cuda()

        mse = nn.MSELoss()

        optim_d_l2h = optim.Adam(filter(lambda p: p.requires_grad, l2h_discriminator.parameters()), lr=learn_rate, betas=(0.0, 0.9))
        optim_g_l2h = optim.Adam(l2h_generator.parameters(), lr=learn_rate, betas=(0.0, 0.9))

        dataloader = LowToHighDataLoader().dataloader

        generator_losses = []
        discriminator_losses = []
        learning_rates = []

        #l2h_generator.load_state_dict(torch.load('l2h_generator_weights.pth'))
        #l2h_discriminator.load_state_dict(torch.load('l2h_discriminator_weights.pth'))

        for epoch in range(max_epoch):
            print("Epoch: ", epoch)
            l2h_generator.train()
            l2h_discriminator.train()

            sum_loss_g = 0
            sum_loss_d = 0
            count_batches = 0
            for i, batch in enumerate(dataloader):
                optim_g_l2h.zero_grad()
                optim_d_l2h.zero_grad()

                x = batch[0].cuda()
                noise = batch[2].cuda()
                y = batch[1].cuda()

                x_reduced = batch[3].cuda()
                hr_gen = l2h_generator(x)
                hr_gen_detach = hr_gen.detach()

                loss_d_l2h = nn.ReLU()(1.0 - l2h_discriminator(y)).mean() + nn.ReLU()(1 + l2h_discriminator(hr_gen_detach)).mean()
                loss_d_l2h.backward()
                optim_d_l2h.step()

                optim_d_l2h.zero_grad()
                gan_loss_h2l = -l2h_discriminator(hr_gen).mean()
                mse_loss_h2l = mse(torchvision.transforms.Resize(size=64)(hr_gen), y)

                loss_g_l2h = alpha * mse_loss_h2l + beta * gan_loss_h2l
                loss_g_l2h.backward()
                optim_g_l2h.step()
                print("\r {}({}) G_l2h: {:.3f}, D_l2h: {:.3f}".format(i + 1, epoch, loss_g_l2h.item(), loss_d_l2h.item()), end='', flush=True)
                sum_loss_d += loss_d_l2h.item()
                sum_loss_g += loss_g_l2h.item()
                count_batches += 1

            avg_loss_g = sum_loss_g / count_batches
            avg_loss_d = sum_loss_d / count_batches
            generator_losses.append(avg_loss_g)
            discriminator_losses.append(avg_loss_d)
            print("Average G_l2h: {:.3f}, D_l2h: {:.3f}".format(avg_loss_g, avg_loss_d))
            self.plot_losses(epoch + 1, generator_losses, discriminator_losses, learning_rates)

            l2h_generator.eval()
            l2h_discriminator.eval()
            torch.save(l2h_generator.state_dict(), 'l2h_generator_weights.pth')
            torch.save(l2h_discriminator.state_dict(), 'l2h_discriminator_weights.pth')


if __name__ == '__main__':
    h2l = LowToHigh()
    h2l.run()