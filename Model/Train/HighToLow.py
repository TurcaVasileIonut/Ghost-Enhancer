import random

import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn, optim

from Data.HighToLowDataLoader import HighToLowDataLoader
from Data.CustomDataset import CustomDataset
from ModelSuperResolution.Discriminator import Discriminator
from ModelSuperResolution.HighToLowGenerator import HighToLowGenerator


class HighToLow:
    def plot_result(self, dataloader, h2l_generator):
        for i, batch in enumerate(dataloader.dataloader):
            plt.subplot(1, 2, 1)
            input_img_tensor = dataloader.denormalize(batch[0][0]).permute(1, 2, 0)
            input_img = input_img_tensor.numpy()
            input_img = (input_img * 255).astype(np.uint8)  # Scale and convert to uint8
            plt.imshow(input_img)
            plt.title('Input Image')
            plt.axis('off')

            x = batch[0].cuda()
            noise = batch[2].cuda()
            lr_gen = h2l_generator(x, noise)
            lr_gen_detach = lr_gen.detach().cpu()

            plt.subplot(1, 2, 2)
            lr_img_tensor = dataloader.denormalize(lr_gen_detach[0]).permute(1, 2, 0)
            lr_img = lr_img_tensor.numpy()
            lr_img = (lr_img * 255).astype(np.uint8)
            plt.imshow(lr_img)
            plt.title('LOW Resolution Image')
            plt.axis('off')

            plt.show()

            cv2.imwrite("../fid/dir1/input_image.png", cv2.cvtColor(input_img, cv2.COLOR_RGBA2BGR))
            cv2.imwrite("../fid/dir2/low_res_image.png",  cv2.cvtColor(lr_img, cv2.COLOR_RGBA2BGR))
            break

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

    def run(self):
        seed_num = 2020
        random.seed(seed_num)
        np.random.seed(seed_num)
        torch.manual_seed(seed_num)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        max_epoch = 5000
        learn_rate = 1e-7
        alpha, beta = 1, 0.05

        h2l_generator = HighToLowGenerator().cuda()
        h2l_discriminator = Discriminator(16).cuda()

        mse = nn.MSELoss()

        optim_d_h2l = optim.Adam(filter(lambda p: p.requires_grad, h2l_discriminator.parameters()), lr=learn_rate, betas=(0.0, 0.9))
        optim_g_h2l = optim.Adam(h2l_generator.parameters(), lr=learn_rate, betas=(0.0, 0.9))

        dataloader = HighToLowDataLoader().dataloader

        h2l_generator.load_state_dict(torch.load('h2l_generator_weights_17.pth'))
        h2l_discriminator.load_state_dict(torch.load('h2l_discriminator_weights_17.pth'))

        #self.plot_result(HighToLowDataLoader(), h2l_generator)

        generator_losses = []
        discriminator_losses = []
        learning_rates = []

        for epoch in range(max_epoch):
            print("Epoch: ", epoch)
            h2l_generator.train()
            h2l_discriminator.train()

            sum_loss_g = 0
            sum_loss_d = 0
            count_batches = 0
            for i, batch in enumerate(dataloader):
                optim_g_h2l.zero_grad()
                optim_d_h2l.zero_grad()

                x = batch[0].cuda()
                y = batch[1].cuda()
                noise = batch[2].cuda()
                x_reduced = batch[3].cuda()
                lr_gen = h2l_generator(x, noise)
                lr_gen_detach = lr_gen.detach()

                loss_d_h2l = nn.ReLU()(1.0 - h2l_discriminator(y)).mean() + nn.ReLU()(1 + h2l_discriminator(lr_gen_detach)).mean()
                loss_d_h2l.backward()
                optim_d_h2l.step()

                optim_d_h2l.zero_grad()
                gan_loss_h2l = -h2l_discriminator(lr_gen).mean()
                #print(lr_gen.shape, x_reduced.shape)
                mse_loss_h2l = mse(lr_gen, x_reduced)

                loss_g_h2l = alpha * mse_loss_h2l + beta * gan_loss_h2l
                loss_g_h2l.backward()
                optim_g_h2l.step()
                print("\r {}({}) G_h2l: {:.3f}, D_h2l: {:.3f}".format(i + 1, epoch, loss_g_h2l.item(), loss_d_h2l.item()), end='', flush=True)
                sum_loss_d += loss_d_h2l.item()
                sum_loss_g += loss_g_h2l.item()
                count_batches += 1

            avg_loss_g = sum_loss_g / count_batches
            avg_loss_d = sum_loss_d / count_batches
            generator_losses.append(avg_loss_g)
            discriminator_losses.append(avg_loss_d)
            print("Average G_h2l: {:.3f}, D_h2l: {:.3f}".format(avg_loss_g, avg_loss_d))
            self.plot_losses(epoch + 1, generator_losses, discriminator_losses, learning_rates)

            h2l_generator.eval()
            h2l_discriminator.eval()
            torch.save(h2l_generator.state_dict(), 'h2l_generator_weights_17.pth')
            torch.save(h2l_discriminator.state_dict(), 'h2l_discriminator_weights_17.pth')


if __name__ == '__main__':
    h2l = HighToLow()
    h2l.run()
