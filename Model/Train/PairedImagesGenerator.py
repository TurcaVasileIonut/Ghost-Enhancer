import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
import os

from Data.HighToLowDataLoader import HighToLowDataLoader
from Data.CustomDataset import CustomDataset
from ModelSuperResolution.HighToLowGenerator import HighToLowGenerator


def run():
    dataloader = HighToLowDataLoader().dataloader

    h2l_generator = HighToLowGenerator().cuda()
    h2l_generator.load_state_dict(torch.load('h2l_generator_weights_17.pth'))
    index = 0
    h2l_generator.eval()

    for i, batch in enumerate(dataloader):
        y_batch = batch[0].cuda()
        noise = batch[2].cuda()
        x_batch = h2l_generator(y_batch, noise)
        x_reduced = batch[3].cuda()

        #print(x_batch.shape)
        #print(y_batch.shape)

        for j in range(x_batch.size(0)):
            single_x = x_batch[j].cpu().detach()
            single_y = y_batch[j].cpu().detach()
            #print(single_x.shape)
            #print(single_y.shape)

            input_img_tensor = single_x.permute(1, 2, 0).detach()
            input_img = input_img_tensor.numpy()
            input_img = (input_img * 255).astype(np.uint8)

            lr_img_tensor = single_y.permute(1, 2, 0).detach()
            lr_img = lr_img_tensor.numpy()
            lr_img = (lr_img * 255).astype(np.uint8)

            '''plt.subplot(1, 2, 1)
            # Scale and convert to uint8
            plt.imshow(input_img)
            plt.title('Input Image')
            plt.axis('off')

            plt.subplot(1, 2, 2)
            plt.imshow(lr_img)
            plt.title('High Resolution Image')
            plt.axis('off')

            plt.show()'''

            # Determine folder based on index
            folder_name = str(index // 1000)  # This will create a new folder name for every 1000 photos
            low_res_path = os.path.join("LowToHighDataset", "LOW", folder_name)
            high_res_path = os.path.join("LowToHighDataset", "HIGH", folder_name)

            # Make sure the directories exist
            os.makedirs(low_res_path, exist_ok=True)
            os.makedirs(high_res_path, exist_ok=True)

            # Save images to their respective folder
            cv2.imwrite(os.path.join(low_res_path, f"{index}.png"), cv2.cvtColor(input_img, cv2.COLOR_RGBA2BGR))
            cv2.imwrite(os.path.join(high_res_path, f"{index}.png"), cv2.cvtColor(lr_img, cv2.COLOR_RGBA2BGR))
            index += 1


if __name__ == '__main__':
    run()
