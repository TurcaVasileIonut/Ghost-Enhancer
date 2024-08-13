import cv2
import numpy as np
import torch
from torch import nn

from ModelSuperResolution.BasicBlock import BasicBlock
from ModelSuperResolution.GhostNet import GhostModule
from ModelSuperResolution.HighToLowGenerator import HighToLowGenerator


class LowToHighGenerator(nn.Module):
    def __init__(self):
        super(LowToHighGenerator, self).__init__()

        size_units = [256, 128, 96]
        count_units = [22, 3, 3]
        inp_res_units = [
            [256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256,
             256], [256, 128, 128], [128, 96, 96]]

        self.a1 = nn.Conv2d(256, 128, kernel_size=1, stride=1)
        self.a2 = nn.Conv2d(128, 96, kernel_size=1, stride=1)

        self.in_layer = nn.Conv2d(3, 256, kernel_size=3, stride=1, padding=1, bias=False)

        self.layers_set = nn.ModuleList()
        self.layers_set_upsampling = nn.ModuleList()
        for i in range(3):
            '''self.layers_set.append(nn.ModuleList())
            self.layers_set_upsampling.append(nn.ModuleList())

            layers = []
            for j in range(count_units[i]):
                if i > 0 and j == 0:
                    layers.append(BasicBlock(size_units[i - 1], size_units[i]))
                else:
                    layers.append(BasicBlock(size_units[i], size_units[i]))

            self.layers_set[i] = nn.Sequential(*layers)
            self.layers_set_upsampling[i] = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.BatchNorm2d(size_units[i]),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(size_units[i], size_units[i], kernel_size=1, stride=1),
            )'''

            nunits = size_units[i]
            current_input_size = inp_res_units[i]

            self.layers_set.append(nn.ModuleList())
            self.layers_set_upsampling.append(nn.ModuleList())

            if i == 0:
                num_blocks_level = 12
            else:
                num_blocks_level = 3

            layers = []
            for j in range(num_blocks_level):
                layers.append(BasicBlock(current_input_size[j], nunits))
                #layers.append(GhostModule(current_input_size[j], nunits))
            self.layers_set[i] = nn.Sequential(*layers)

            self.layers_set_upsampling[i] = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.BatchNorm2d(size_units[i]),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(size_units[i], size_units[i], kernel_size=1, stride=1),
            )

        self.output_layer = nn.Sequential(
            nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),

            nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),

            nn.Conv2d(96, 3, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        out = self.in_layer(input)
        for i in range(3):
            temp = self.layers_set[i](out)
            if i == 0:
                out = out + temp
            elif i == 1:
                temp2 = self.a1(out)
                out = temp + temp2
            elif i == 2:
                temp2 = self.a2(out)
                out = temp + temp2
            if i < 2:
                out = self.layers_set_upsampling[i](out)
        out = self.output_layer(out)
        return out


if __name__ == "__main__":
    net = LowToHighGenerator().cuda()
    X = np.random.randn(1, 3, 16, 16).astype(np.float32)
    X = torch.from_numpy(X).cuda()
    Y = net(X)
    print(Y.shape)

    Xim = X.cpu().numpy().squeeze().transpose(1,2,0)
    Yim = Y.detach().cpu().numpy().squeeze().transpose(1,2,0)
    Xim = (Xim - Xim.min()) / (Xim.max() - Xim.min())
    Yim = (Yim - Yim.min()) / (Yim.max() - Yim.min())
    cv2.imshow("X", Xim)
    cv2.imshow("Y", Yim)
    cv2.waitKey()
    print("finished.")
