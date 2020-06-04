#!/usr/bin/python3

#original algorithm committed by Erik Lindner-Noren
#see https://github.com/eriklindernoren/PyTorch-Deep-Dream

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import models,transforms
import numpy as np
from PIL import Image
import argparse
import os
import tqdm
import scipy.ndimage as nd
import imageio
from utils import *

def dream(image, model, iterations, lr):
    """ Updates the image to maximize outputs for n iterations """
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available else torch.FloatTensor
    image = Variable(Tensor(image), requires_grad=True)
    for i in range(iterations):
        model.zero_grad()
        out = model(image)
        loss = out.norm()
        loss.backward()
        avg_grad = np.abs(image.grad.data.cpu().numpy()).mean()
        norm_lr = lr / avg_grad
        image.data += norm_lr * image.grad.data
        image.data = clip(image.data)
        image.grad.data.zero_()
    return image.cpu().data.numpy()

def deep_dream(image, model, iterations, lr, octave_scale, num_octaves):
    """ Main deep dream method """
    image = preprocess(image).unsqueeze(0).cpu().data.numpy()
    # Extract image representations for each octave
    octaves = [image]
    for _ in range(num_octaves - 1):
        octaves.append(nd.zoom(octaves[-1], (1, 1, 1 / octave_scale, 1 / octave_scale), order=1))
    detail = np.zeros_like(octaves[-1])
    for octave, octave_base in enumerate(tqdm.tqdm(octaves[::-1], desc="Dreaming")):
        if octave > 0:
            # Upsample detail to new octave dimension
            detail = nd.zoom(detail, np.array(octave_base.shape) / np.array(detail.shape), order=1)
        # Add deep dream detail from previous octave to new base
        input_image = octave_base + detail
        # Get new deep dream image
        dreamed_image = dream(input_image, model, iterations, lr)
        # Extract deep dream details
        detail = dreamed_image - octave_base
    return deprocess(dreamed_image)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_image", type=str, default="bild01.jpg", help="path to input image")
    parser.add_argument("--iterations", default=25,type=int, help="number of gradient ascent steps per octave")
    parser.add_argument("--at_layer", default=27, type=int, help="layer at which we modify image to maximize outputs")
    parser.add_argument("--lr", default=0.01, type=float, help="learning rate")
    parser.add_argument("--octave_scale", default=1.4,type=float, help="image scale between octaves")
    parser.add_argument("--num_octaves", default=10,type=int, help="number of octaves")
    parser.add_argument("--target_folder", type=str, default="StyledImage")
    args = parser.parse_args()

    # Load image
    image = Image.open(args.input_image).convert("RGB")
    w,h = image.size
    verh = float(h/w)
    resize = False
    while w*h > 1280*720:
       resize = True
       if h/w <= verh:
         w -= 1
       elif h/w > verh:
         h -= 1
    if resize:
       resizer = transforms.Compose([transforms.Resize((int(h),int(w)))])
       image = resizer(image)

    network = models.vgg19(pretrained=True)
    layers = list(network.features.children())
    model = nn.Sequential(*layers[: (args.at_layer + 1)])
    if torch.cuda.is_available:
        model = model.cuda()

    dreamed_image = deep_dream(image, model,
          iterations=args.iterations,
          lr=args.lr,
          octave_scale=args.octave_scale,
          num_octaves=args.num_octaves,)

    image_number = 0
    target_folder = args.target_folder
    while os.path.exists((target_folder+"/Image"+(4-len(str(image_number)))*"0"+str(image_number)+".jpg")):
        image_number += 1
    pfad = target_folder+"/Image"+(4-len(str(image_number)))*"0"+str(image_number)+".jpg"
    imageio.imwrite(pfad,dreamed_image)

