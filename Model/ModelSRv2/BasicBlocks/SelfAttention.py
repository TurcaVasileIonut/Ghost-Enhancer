import torch
import torch.nn as nn

from ModelSRv2.Utils.ImagePlotter import ImagePlotter


class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=embed_size, num_heads=heads)

    def forward(self, x):
        # Assuming x is of shape [batch_size, channels, height, width]
        batch_size, channels, height, width = x.size()

        # Reshape x to [batch_size, height * width, channels] for the attention layer
        x = x.view(batch_size, channels, height * width).permute(0, 2, 1)

        # Apply self-attention
        attn_output, _ = self.attention(x, x, x)

        # Optionally reshape or process the output further
        attn_output = attn_output.permute(0, 2, 1).view(batch_size, channels, height, width)

        return attn_output


if __name__ == '__main__':
    model = SelfAttention(embed_size=64, heads=16)
    image = torch.randn(3, 64, 16, 16)  # Example single image

    output = model(image)
    print(output.shape)
    # ImagePlotter.plot_images(image[0], output[0])
