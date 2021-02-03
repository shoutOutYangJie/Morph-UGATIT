import random
import numpy as np
import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
from abc import ABC, abstractmethod

def get_transform(opt, params=None, grayscale=False, method=Image.BILINEAR):
    transform_list = []
    if grayscale:
        transform_list.append(transforms.Grayscale(1))

    osize = [opt.load_size, opt.load_size]
    transform_list.append(transforms.Resize(osize, method))

    # transform_list.append(transforms.RandomCrop(opt.crop_size))

    # if opt.preprocess == 'none':
    #     transform_list.append(transforms.Lambda(lambda img: __make_power_2(img, base=4, method=method)))

    transform_list.append(transforms.RandomHorizontalFlip())

    transform_list += [transforms.ToTensor()]
    if grayscale:
        transform_list += [transforms.Normalize((0.5,), (0.5,))]
    else:
        transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]  # -1, 1
    return transforms.Compose(transform_list)

def get_transform_for_anime(opt, params=None, grayscale=False, method=Image.BILINEAR):
    transform_list = []
    if grayscale:
        transform_list.append(transforms.Grayscale(1))

    osize = [opt.load_size+30, opt.load_size+30]
    transform_list.append(transforms.Resize(osize, method))

    transform_list.append(transforms.RandomHorizontalFlip())

    transform_list.append(transforms.RandomCrop(opt.load_size))

    transform_list += [transforms.ToTensor()]
    if grayscale:
        transform_list += [transforms.Normalize((0.5,), (0.5,))]
    else:
        transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]  # -1, 1
    return transforms.Compose(transform_list)

def __make_power_2(img, base, method=Image.BICUBIC):
    ow, oh = img.size
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    if h == oh and w == ow:
        return img

    __print_size_warning(ow, oh, w, h)
    return img.resize((w, h), method)



def __crop(img, pos, size):
    ow, oh = img.size
    x1, y1 = pos
    tw = th = size
    if (ow > tw or oh > th):
        return img.crop((x1, y1, x1 + tw, y1 + th))
    return img


def __flip(img, flip):
    if flip:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img


def __print_size_warning(ow, oh, w, h):
    """Print warning information about image size(only print once)"""
    if not hasattr(__print_size_warning, 'has_printed'):
        print("The image size needs to be a multiple of 4. "
              "The loaded image size was (%d, %d), so it was adjusted to "
              "(%d, %d). This adjustment will be done to all images "
              "whose sizes are not multiples of 4" % (ow, oh, w, h))
        __print_size_warning.has_printed = True