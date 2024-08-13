import os
import random

import cv2
from matplotlib import pyplot as plt

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch


class CustomDataset(Dataset):
    def __init__(self, X_data, Y_data):
        self.X = [os.path.join(d, i) for d in X_data for i in os.listdir(d) if os.path.isfile(os.path.join(d, i))]
        self.Y = [os.path.join(d, i) for d in Y_data for i in os.listdir(d) if os.path.isfile(os.path.join(d, i))]

        self.lr_idx = 0
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        input = cv2.imread(self.X[index])
        input = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)
        input = self.transforms(input)

        y_index = index
        if len(self.X) > len(self.Y):
            y_index = random.randint(0, len(self.Y) - 1)

        output = cv2.imread(self.Y[y_index])
        output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
        output = self.transforms(output)
        noise = torch.randn(1, 64, dtype=torch.float32)
        reduced = torch.nn.functional.avg_pool2d(input, 4, 4)
        return [input, output, noise, reduced]

    def get_noise(self, n):
        return torch.randn(n, 1, 64, dtype=torch.float32)

    def denormalize(self, tensor):
        means = [0.5, 0.5, 0.5]
        stds = [0.5, 0.5, 0.5]
        denormalized = tensor.clone()
        for t, m, s in zip(denormalized, means, stds):
            t.mul_(s).add_(m)
        return denormalized


if __name__ == "__main__":
    dataset = CustomDataset()
    sample = dataset.__getitem__(0)
    plt.figure(figsize=(8, 4))

    print(sample[0].shape)
    plt.subplot(1, 2, 1)
    plt.imshow(dataset.denormalize(sample[0]).permute(1, 2, 0))
    plt.title('Input Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(dataset.denormalize(sample[1]).permute(1, 2, 0))
    plt.title('LOW Resolution Image')
    plt.axis('off')

    plt.show()