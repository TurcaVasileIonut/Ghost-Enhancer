import cv2
import numpy as np
import torch
from torch import nn

from ModelSuperResolution.BasicBlock import BasicBlock


class HighToLowGenerator(nn.Module):
    def __init__(self):
        super(HighToLowGenerator, self).__init__()
        blocks = [96, 96, 128, 128, 256, 256, 512, 512, 128, 128, 32, 32]

        self.in_layer = nn.Conv2d(4, blocks[0], kernel_size=3, stride=1, padding=1, bias=False)

        self.downsample = nn.Sequential(
            BasicBlock(blocks[0], blocks[0], downsample=True),
            BasicBlock(blocks[0], blocks[1]),

            BasicBlock(blocks[1], blocks[2], downsample=True),
            BasicBlock(blocks[2], blocks[3]),

            BasicBlock(blocks[3], blocks[4], downsample=True),
            BasicBlock(blocks[4], blocks[5]),

            BasicBlock(blocks[5], blocks[6], downsample=True),
            BasicBlock(blocks[6], blocks[7]),
        )

        self.upsample = nn.Sequential(
            nn.PixelShuffle(2),
            BasicBlock(blocks[8], blocks[8]),
            BasicBlock(blocks[9], blocks[9]),
            nn.PixelShuffle(2),
            BasicBlock(blocks[10], blocks[10]),
            BasicBlock(blocks[11], blocks[11]),
        )

        self.out_layer = nn.Sequential(
            nn.Conv2d(blocks[-1], 8, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(8, 3, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Tanh()
        )

        self.noise = nn.Linear(64, 4096)

    def forward(self, x, z=None):
        if z is None:
            z = np.random.randn(1, 1, 64).astype(np.float32)
            z = torch.from_numpy(z).cuda()
        noises = self.noise(z)
        noises = noises.view(-1, 1, 64, 64)
        out = torch.cat((x, noises), dim=1)
        out = self.in_layer(out)
        out = self.downsample(out)
        out = self.upsample(out)
        out = self.out_layer(out)
        return out


def high2low_test():
    net = HighToLowGenerator().cuda()
    X = np.random.randn(1, 3, 64, 64).astype(np.float32)
    X = torch.from_numpy(X).cuda()
    Y = net(X)
    print(Y.shape)
    Xim = X.cpu().numpy().squeeze().transpose(1, 2, 0)
    Yim = Y.detach().cpu().numpy().squeeze().transpose(1, 2, 0)
    Xim = (Xim - Xim.min()) / (Xim.max() - Xim.min())
    Yim = (Yim - Yim.min()) / (Yim.max() - Yim.min())
    cv2.imshow("X", Xim)
    cv2.imshow("Y", Yim)
    cv2.waitKey()


if __name__ == "__main__":
    high2low_test()
