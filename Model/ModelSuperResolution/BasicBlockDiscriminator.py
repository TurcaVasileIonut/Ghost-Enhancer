from torch import nn


class BasicBlockDiscriminator(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=False, nobn=False):
        super(BasicBlockDiscriminator, self).__init__()
        self.downsample = downsample
        self.nobn = nobn

        self.conv1 = nn.utils.spectral_norm(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False))
        if not self.nobn:
            self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=False)

        if self.downsample:
            self.conv2 = nn.Sequential(
                nn.AvgPool2d(2, 2),
                nn.utils.spectral_norm(
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)))
        else:
            self.conv2 = nn.utils.spectral_norm(
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False))

        if not self.nobn:
            self.bn2 = nn.BatchNorm2d(out_channels)
        if in_channels != out_channels or self.downsample:
            if self.downsample:
                self.skip = nn.Sequential(
                    nn.AvgPool2d(2, 2),
                    nn.utils.spectral_norm(nn.Conv2d(in_channels, out_channels, 1, 1)))
            else:
                self.skip = nn.utils.spectral_norm(nn.Conv2d(in_channels, out_channels, 1, 1, 0))
        else:
            self.skip = None
        self.stride = stride

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