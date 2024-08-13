import os
import random

import cv2
from PIL import Image
from matplotlib import pyplot as plt

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch


class CustomDatasetMathRes(Dataset):
    def __init__(self, Data):
        self.X = [os.path.join(d, i) for d in Data for i in os.listdir(d) if os.path.isfile(os.path.join(d, i))]

        self.lr_idx = 0
        self.lr_transform = transforms.Compose(
            [
                transforms.Resize((64 // 4, 64 // 4), Image.BICUBIC),
                transforms.ToTensor()
            ]
        )

        self.hr_transform = transforms.Compose(
            [
                transforms.ToTensor()
            ]
        )

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        output = cv2.imread(self.X[index])
        output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
        output = Image.fromarray(output)
        input = self.lr_transform(output)
        output = self.hr_transform(output)
        return [input, output]


if __name__ == "__main__":
    dataset = CustomDatasetMathRes(["../Data/LowToHighDataset/HIGH/0"])
    sample = dataset.__getitem__(0)
    plt.figure(figsize=(8, 4))

    print(sample[0].shape)
    plt.subplot(1, 2, 1)
    plt.imshow(sample[0].permute(1, 2, 0))
    plt.title('Input Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(sample[1].permute(1, 2, 0))
    plt.title('LOW Resolution Image')
    plt.axis('off')

    plt.show()