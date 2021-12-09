import torch
import numpy as np
from torch import nn
from einops.layers.torch import Rearrange

import math

def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()

        self.net1 = nn.Linear(dim, hidden_dim)
        self.net2 = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = self.net1(x)
        x = gelu(x)
        return self.net2(x)

class MixerBlock(nn.Module):
    def __init__(self, dim, num_patch, token_dim, channel_dim, dropout = 0.):
        super().__init__()

        self.token_mix1 = nn.Sequential(
            nn.LayerNorm(dim),
            Rearrange('b n d -> b d n'),
            FeedForward(num_patch, token_dim, dropout),
            Rearrange('b d n -> b n d')
        )
        self.token_mix2 = nn.Sequential(
            nn.LayerNorm(dim),
            Rearrange('b n d -> b d n'),
            FeedForward(num_patch, token_dim, dropout),
            Rearrange('b d n -> b n d')
        )

        self.channel_mix1 = nn.Sequential(
            nn.LayerNorm(dim),
            FeedForward(dim, channel_dim, dropout)
        )
        self.channel_mix2 = nn.Sequential(
            nn.LayerNorm(dim),
            FeedForward(dim, channel_dim, dropout)
        )

    def forward(self, x1, x2):

        x1 = x1 + self.token_mix1(x1)
        x2 = x2 + self.token_mix2(x2)

        x1_out = x1 + self.channel_mix1(x1+x2)
        x2_out = x2 + self.channel_mix2(x2+x1)

        return x1_out, x2_out


class MLPMixer(nn.Module):
    def __init__(self, in_chan, dim, patch_size, image_size, depth, token_dim, channel_dim):
        super(MLPMixer, self).__init__()
        self.to_patch_embedding1 = nn.Sequential(
            nn.Conv2d(in_channels=in_chan, out_channels=dim, kernel_size=patch_size, stride=patch_size),
            Rearrange('b c h w -> b (h w) c')
        )
        self.to_patch_embedding2 = nn.Sequential(
            nn.Conv2d(in_channels=in_chan, out_channels=dim, kernel_size=patch_size, stride=patch_size),
            Rearrange('b c h w -> b (h w) c')
        )
        self.anti_patch_embedding1 = nn.Sequential(
            Rearrange('b (h w) c -> b c h w', h=8, w=8),
            nn.ConvTranspose2d(128, 128, kernel_size=4, stride=4),
            nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        )
        self.anti_patch_embedding2 = nn.Sequential(
            Rearrange('b (h w) c -> b c h w', h=8, w=8),
            nn.ConvTranspose2d(128, 128, kernel_size=4, stride=4),
            nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        )

        self.mixer_blocks = nn.ModuleList([])

        for _ in range(depth):
            self.mixer_blocks.append(MixerBlock(dim, 64, token_dim, channel_dim))

        self.layer_norm1 = nn.LayerNorm(dim)
        self.layer_norm2 = nn.LayerNorm(dim)
    
        self.conv_mix = nn.Conv2d(256, 256, kernel_size=1)

    def forward(self, x1, x2):
        b, c, h, w = x1.size() # [B, 256, 64, 64]

        # Step1: Patch_embedding   [B, 8*8, dim=128]
        x1 = self.to_patch_embedding1(x1)
        x2 = self.to_patch_embedding2(x2)
        x1_patch, x2_patch = x1, x2
        # Step2: Cross-Mixer
        for mixer_block in self.mixer_blocks:
            x1_patch, x2_patch = mixer_block(x1_patch, x2_patch)    # [B, 64, 128]
                            
        x1_out = self.layer_norm1(x1_patch)                  # [B, 64, 128]
        x2_out = self.layer_norm2(x2_patch)

        # Step3: skip connection     out_size [B, 64, 128]              
        x1_out = x1 + x1_out
        x2_out = x2 + x2_out

        # Step4: recover to 4 dimmension
        # B, 64, 128 -> B, 128, 8, 8 -> B 128, 32, 32
        #                            -> B, 128, 64, 64
        x1_out = self.anti_patch_embedding1(x1_out)
        x2_out = self.anti_patch_embedding1(x2_out)
        # Step5: cat and adjust
        out = torch.cat((x1_out, x2_out), 1)               # [B, 256, 64, 64]

        out = self.conv_mix(out)

        return out


if __name__ == "__main__":

    img1 = torch.ones([1, 256, 64, 64])
    img2 = torch.ones([1, 256, 64, 64])


    # model = MLPMixer(in_channels=3, image_size=256, patch_size=16, num_classes=1000,
                    #  dim=512, depth=8, token_dim=256, channel_dim=2048)

    model = MLPMixer(in_chan=256, dim=128, patch_size=8, image_size=64, depth=7, token_dim=512, channel_dim=1024)

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
    print('Trainable Parameters: %.3fM' % parameters)

    out_img = model(img1, img2)

    print("Shape of out :", out_img.shape)  # [B, in_channels, image_size, image_size]




