import numpy as np
from torch import nn
import torch
from torch.nn.parameter import Parameter

from ModelSuperResolution.BasicBlockDiscriminator import BasicBlockDiscriminator


class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()

        self.blocks = [128, 128, 256, 256, 512, 512]

        self.out_layer = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(16 * self.blocks[-1], self.blocks[-1])),
            nn.ReLU(),
            nn.utils.spectral_norm(nn.Linear(self.blocks[-1], 1))
        )

        self.residual_blocks = nn.Sequential(
            BasicBlockDiscriminator(3, self.blocks[0], nobn=True),
            BasicBlockDiscriminator(self.blocks[0], self.blocks[1], nobn=True),
            BasicBlockDiscriminator(self.blocks[1], self.blocks[2], downsample=(input_dim == 64), nobn=True),
            BasicBlockDiscriminator(self.blocks[2], self.blocks[3], downsample=(input_dim == 64), nobn=True),
            BasicBlockDiscriminator(self.blocks[3], self.blocks[4], downsample=True, nobn=True),
            BasicBlockDiscriminator(self.blocks[4], self.blocks[5], downsample=True, nobn=True),
        )

    def forward(self, x):
        out = self.residual_blocks(x)
        out = out.view(-1, 16 * self.blocks[-1])
        out = self.out_layer(out)
        return out


def discriminator_test():
    in_size = 64
    net = Discriminator(in_size).cuda()
    X = np.random.randn(2, 3, in_size, in_size).astype(np.float32)
    X = torch.from_numpy(X).cuda()
    Y = net(X)
    print(Y.shape)


if __name__ == '__main__':
    discriminator_test()
