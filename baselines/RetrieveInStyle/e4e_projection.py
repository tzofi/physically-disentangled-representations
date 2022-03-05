import os
import sys
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
from argparse import Namespace
from e4e.models.psp import pSp
from util import *


@ torch.no_grad()
def projection(img, net, device='cuda'):

    images, w_plus = net(img, randomize_noise=False, return_latents=True)
    result_file = {}
    return w_plus

@ torch.no_grad()
def extract_projection(img, net, device='cuda'):

    transform = transforms.Compose(
        [
            transforms.Resize(64),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    img = transform(img).unsqueeze(0).to(device)
    images, w_plus = net(img, randomize_noise=False, return_latents=True)
    result_file = {}
    return w_plus
