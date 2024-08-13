from torch import nn


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=False, upsample=False, nobn=False):
        super(BasicBlock, self).__init__()

        self.downsample = downsample
        self.upsample = upsample
        self.nobn = nobn
        self.stride = stride

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

        if self.upsample:
            self.conv1 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)

        if self.downsample:
            self.conv2 = nn.Sequential(
                nn.AvgPool2d(kernel_size=2, stride=2),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False))
        else:
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        
        if in_channels != out_channels or self.upsample or self.downsample:
            if self.upsample:
                self.skip = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
            elif self.downsample:
                self.skip = nn.Sequential(
                    nn.AvgPool2d(kernel_size=2, stride=2),
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1))
            else:
                self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        else:
            self.skip = None

    def forward(self, x):
        residual = x
        if not self.nobn:
            out = self.bn1(x)
            out = self.relu(out)
        else:
            out = self.relu(x)
        out = self.conv1(out)
        if not self.nobn:
            out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        if self.skip is not None:
            residual = self.skip(x)
        out += residual
        return out
