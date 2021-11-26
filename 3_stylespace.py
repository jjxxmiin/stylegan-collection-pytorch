import argparse
import math
import os

import torch
import torch.nn.functional as F
import torchvision
from torch import optim
from tqdm import tqdm

from src.models.stylegan import Generator


def conv_warper(layer, input, style, noise):
    """[summary]

    Args:
        layer (nn.Module): StyleConv
        input ([type]): [description]
        style ([type]): [description]
        noise ([type]): [description]
    """
    conv = layer.conv
    batch, in_channel, height, width = input.shape
    
    style = style.view(batch, 1, in_channel, 1, 1)
    weight = conv.scale * conv.weight * style
    
    if conv.demodulate:
        demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8)
        weight = weight * demod.view(batch, conv.out_channel, 1, 1, 1)

    weight = weight.view(
        batch * conv.out_channel, in_channel, conv.kernel_size, conv.kernel_size
    )
    
    if conv.upsample:
        input = input.view(1, batch * in_channel, height, width)
        weight = weight.view(
            batch, conv.out_channel, in_channel, conv.kernel_size, conv.kernel_size
        )
        weight = weight.transpose(1, 2).reshape(
            batch * in_channel, conv.out_channel, conv.kernel_size, conv.kernel_size
        )
        out = F.conv_transpose2d(input, weight, padding=0, stride=2, groups=batch)
        _, _, height, width = out.shape
        out = out.view(batch, conv.out_channel, height, width)
        out = conv.blur(out)

    elif conv.downsample:
        input = conv.blur(input)
        _, _, height, width = input.shape
        input = input.view(1, batch * in_channel, height, width)
        out = F.conv2d(input, weight, padding=0, stride=2, groups=batch)
        _, _, height, width = out.shape
        out = out.view(batch, conv.out_channel, height, width)

    else:
        input = input.view(1, batch * in_channel, height, width)
        out = F.conv2d(input, weight, padding=conv.padding, groups=batch)
        _, _, height, width = out.shape
        out = out.view(batch, conv.out_channel, height, width)

    out = layer.noise(out, noise=noise)
    out = layer.activate(out)
    
    return out
    

def encoder(G, noise):
    styles = [noise]
    style_space = []
    
    styles = [G.style(s) for s in styles]
    noise = [getattr(G.noises, f'noise_{i}') for i in range(G.num_layers)]
    inject_index = G.n_latent
    latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
    style_space.append(G.conv1.conv.modulation(latent[:, 0]))
    
    i = 1
    for conv1, conv2 in zip(G.convs[::2], G.convs[1::2]):
        style_space.append(conv1.conv.modulation(latent[:, i]))
        style_space.append(conv2.conv.modulation(latent[:, i + 1]))
        i += 2
    
    return style_space, latent, noise
    
    
def decoder(G, style_space, latent, noise):
    out = G.input(latent)
    out = conv_warper(G.conv1, out, style_space[0], noise[0])
    skip, _ = G.to_rgb1(out, latent[:, 1])
    
    i = 1
    for conv1, conv2, noise1, noise2, to_rgb in zip(
        G.convs[::2], G.convs[1::2], noise[1::2], noise[2::2], G.to_rgbs
    ):
        out = conv_warper(conv1, out, style_space[i], noise=noise1)
        out = conv_warper(conv2, out, style_space[i+1], noise=noise2)
        skip, _ = to_rgb(out, latent[:, i + 2], skip)
        i += 2

    image = skip

    return image


def main():
    index = [0,1,1,2,2,3,4,4,5,6,6,7,8,8,9,10,10,11,12,12,13,14,14,15,16,16]
    LOAD_PATH = '../checkpoint/stylegan2-ffhq-config-f.pt'

    model = Generator(
        size=1024,
        style_dim=512,
        n_mlp=8,
        channel_multiplier=2
    )

    model.load_state_dict(torch.load(LOAD_PATH)['g_ema'])
    model.eval()
    model.cuda()

    test_input = torch.randn(1,512).cuda()

    output, _ = model([test_input], return_latents=False)

    torchvision.utils.save_image(output.detach().cpu(), 
                                 os.path.join("./results/stylespace_origin.jpg"), 
                                 normalize=True, 
                                 scale_each=True, 
                                 range=(-1, 1))

    style_space, latent, noise = encoder(model, test_input)
    style_space[index[9]][:, 409] += 10
    image = decoder(model, style_space, latent, noise)
    
    torchvision.utils.save_image(image.detach().cpu(), 
                                 os.path.join("./results/stylespace_eye.jpg"), 
                                 normalize=True, 
                                 scale_each=True, 
                                 range=(-1, 1))
    
    style_space, latent, noise = encoder(model, test_input)
    style_space[index[12]][:, 330] -= 50
    image = decoder(model, style_space, latent, noise)
    
    torchvision.utils.save_image(image.detach().cpu(), 
                                 os.path.join("./results/stylespace_hair.jpg"), 
                                 normalize=True, 
                                 scale_each=True, 
                                 range=(-1, 1))
    
    style_space, latent, noise = encoder(model, test_input)
    style_space[index[6]][:, 259] -= 20
    image = decoder(model, style_space, latent, noise)
    
    torchvision.utils.save_image(image.detach().cpu(), 
                                 os.path.join("./results/stylespace_mouth.jpg"), 
                                 normalize=True, 
                                 scale_each=True, 
                                 range=(-1, 1))
    
    style_space, latent, noise = encoder(model, test_input)
    style_space[index[15]][:, 45] -= 3
    image = decoder(model, style_space, latent, noise)
    
    torchvision.utils.save_image(image.detach().cpu(), 
                                 os.path.join("./results/stylespace_lip.jpg"), 
                                 normalize=True, 
                                 scale_each=True, 
                                 range=(-1, 1))
    
    
if __name__ == "__main__":
    main()    




