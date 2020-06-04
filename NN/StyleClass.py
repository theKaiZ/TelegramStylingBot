#!/usr/bin/python3

#Original algorithm developed by Leon A. Gatys, Alexander S. Ecker and Matthias Bethge
#See https://pytorch.org/tutorials/advanced/neural_style_tutorial.html

from __future__ import print_function,division
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
import copy
import os
import argparse

def wandle_Bild_um(content_img,style_img, runs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert style_img.size() == content_img.size(), \
        "we need to import style and content images of the same size"

    class ContentLoss(nn.Module):
        def __init__(self, target,):
            super(ContentLoss, self).__init__()
            # we 'detach' the target content from the tree used
            # to dynamically compute the gradient: this is a stated value,
            # not a variable. Otherwise the forward method of the criterion
            # will throw an error.
            self.target = target.detach()
        def forward(self, input):
            self.loss = F.mse_loss(input, self.target)
            return input

    def gram_matrix(input):
        a, b, c, d = input.size()  # a=batch size(=1)
        # b=number of feature maps
        # (c,d)=dimensions of a f. map (N=c*d)
        features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL
        G = torch.mm(features, features.t())  # compute the gram product
        # we 'normalize' the values of the gram matrix
        # by dividing by the number of element in each feature maps.
        return G.div(a * b * c * d)

    class StyleLoss(nn.Module):

        def __init__(self, target_feature):
            super(StyleLoss, self).__init__()
            self.target = gram_matrix(target_feature).detach()

        def forward(self, input):
            G = gram_matrix(input)
            self.loss = F.mse_loss(G, self.target)
            return input

    cnn = models.vgg19(pretrained=True).features.to(device).eval()
    cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

    class Normalization(nn.Module):
        def __init__(self, mean, std):
            super(Normalization, self).__init__()
            # .view the mean and std to make them [C x 1 x 1] so that they can
            # directly work with image Tensor of shape [B x C x H x W].
            # B is batch size. C is number of channels. H is height and W is width.
            self.mean = mean.clone().detach().view(-1,1,1)#torch.tensor(mean).view(-1, 1, 1)
            self.std = std.clone().detach().view(-1,1,1)#torch.tensor(std).view(-1, 1, 1)

        def forward(self, img):
            # normalize img
            return (img - self.mean) / self.std

    content_layers_default = ['conv_4']
    style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
    def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                                   style_img, content_img,
                                   content_layers=content_layers_default,
                                   style_layers=style_layers_default):
        cnn = copy.deepcopy(cnn)
        # normalization module
        normalization = Normalization(normalization_mean, normalization_std).to(device)
        # just in order to have an iterable access to or list of content/syle
        # losses
        content_losses = []
        style_losses = []
        # assuming that cnn is a nn.Sequential, so we make a new nn.Sequential
        # to put in modules that are supposed to be activated sequentially
        model = nn.Sequential(normalization)
        i = 0  # increment every time we see a conv
        for layer in cnn.children():
            if isinstance(layer, nn.Conv2d):
                i += 1
                name = 'conv_{}'.format(i)
            elif isinstance(layer, nn.ReLU):
                name = 'relu_{}'.format(i)
                # The in-place version doesn't play very nicely with the ContentLoss
                # and StyleLoss we insert below. So we replace with out-of-place
                # ones here.
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                name = 'pool_{}'.format(i)
            elif isinstance(layer, nn.BatchNorm2d):
                name = 'bn_{}'.format(i)
            else:
                raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))
            model.add_module(name, layer)
            if name in content_layers:
                # add content loss:
                target = model(content_img).detach()
                content_loss = ContentLoss(target)
                model.add_module("content_loss_{}".format(i), content_loss)
                content_losses.append(content_loss)
            if name in style_layers:
                # add style loss:
                target_feature = model(style_img).detach()
                style_loss = StyleLoss(target_feature)
                model.add_module("style_loss_{}".format(i), style_loss)
                style_losses.append(style_loss)
        # now we trim off the layers after the last content and style losses
        for i in range(len(model) - 1, -1, -1):
            if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
                break
        model = model[:(i + 1)]
        return model, style_losses, content_losses

    input_img = content_img.clone()

    def get_input_optimizer(input_img):
        # this line to show that input is a parameter that requires a gradient
        optimizer = optim.LBFGS([input_img.requires_grad_()])
        return optimizer

    def run_style_transfer(cnn, normalization_mean, normalization_std,
                           content_img, style_img, input_img, num_steps=300,
                           style_weight=1000000, content_weight=1):
        """Run the style transfer."""
        model, style_losses, content_losses = get_style_model_and_losses(cnn,
                normalization_mean, normalization_std, style_img, content_img)
        optimizer = get_input_optimizer(input_img)
        run = [0]
        while run[0] < num_steps:
            def closure():
                # correct the values of updated input image
                input_img.data.clamp_(0, 1)

                optimizer.zero_grad()
                model(input_img)

                style_score = 0
                content_score = 0

                for sl in style_losses:
                    style_score += sl.loss
                for cl in content_losses:
                    content_score += cl.loss

                style_score *= style_weight
                content_score *= content_weight

                loss = style_score + content_score
                if run[0] < num_steps:
                  loss.backward()

                run[0] += 1
                if run[0] % 50 == 0:
                  print(run[0],"von",num_steps)
                return style_score + content_score
            if run[0] < num_steps:
              optimizer.step(closure)
        return input_img.data.clamp_(0, 1)
    return run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
                                content_img, style_img, input_img, runs)


class StyleMe():
  unloader = transforms.Compose([transforms.ToPILImage()])  # reconvert into PIL image

  def __init__(self,input_image_pfad,output_folder="StyledImages"):
     self.input_image = Image.open(input_image_pfad)
     self.output_folder = output_folder
     self.w,self.h = self.input_image.size
     verh = float(self.h / self.w)
     while self.w*self.h > 1280*720:
        if self.h/self.w <= verh:
           self.w -= 1
        elif self.h/self.w > verh:
           self.h -= 1
     self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
     self.loader = transforms.Compose([
        transforms.Resize((int(self.h),int(self.w))),  # scale imported image
        transforms.ToTensor()])  # transform it into a torch tensor
     self.loader_drehen = transforms.Compose([
        transforms.RandomRotation(90),
        transforms.Resize((int(self.h),int(self.w))),  # scale imported image
        transforms.ToTensor()])
    
  def image_loader(self,image):
      image = self.loader(image).unsqueeze(0)
      return image.to(self.device, torch.float)

  def image_loader_drehen(self,image):
      image = self.loader_drehen(image).unsqueeze(0)
      return image.to(self.device, torch.float)

  def run(self, style_image_pfad, steps):
     self.input_image = self.image_loader(self.input_image)
     self.style_image = Image.open(style_image_pfad)
     style_w,style_h = self.style_image.size
     if (self.w > self.h and style_w > style_h) or (self.w < self.h and style_w < style_h):
         self.style_image = self.image_loader(self.style_image)
     else:
         self.style_image = self.image_loader_drehen(self.style_image)
     self.input_image = self.unloader(wandle_Bild_um(self.input_image,self.style_image,steps)[0].to("cpu"))

  def save_image(self) :
     image_number = 0
     while os.path.exists((self.output_folder+"Image"+(4-len(str(image_number)))*"0"+str(image_number)+".jpg")):
         image_number += 1
     pfad = self.output_folder+"Image"+(4-len(str(image_number)))*"0"+str(image_number)+".jpg"
     self.input_image.save(pfad)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_image", type=str, default="image0007.jpg", help="path to input image")
    parser.add_argument("--style_image",type=str, default="Stylesammlung/[Frak].jpg")
    parser.add_argument("--target_folder", type=str, default="StyledImage/")
    parser.add_argument("--iterations", type=int, default=250)
    args = parser.parse_args()
    Transfer = StyleMe(args.input_image,args.target_folder)
    Transfer.run(args.style_image,args.iterations)
    Transfer.save_image()

