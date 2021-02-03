import torch
from enum import IntEnum
from skimage import io
import numpy as np
from .utils import crop, load_image, get_preds_fromhm, draw_gaussian, recreate_aligned_images
from matplotlib import pyplot as plt
import os
import cv2
import dlib


class FaceAlignment:
    def __init__(self, dlib_model):
        self.face_regressor = dlib.shape_predictor(dlib_model)
        self.face_detector = dlib.get_frontal_face_detector()

    def get_landmarks_from_image(self, image):
        rect = self.face_detector(image, 1)  # 只要一个人脸
        if len(rect) == 0:
            return None
        rect = rect[0]
        pts = self.face_regressor(image, rect).parts()
        pts = np.array([[pt.x, pt.y] for pt in pts]) # 68, 2
        return pts

    def rotate_align_crop(self, image, size):
        pts = self.get_landmarks_from_image(image)
        if pts is None:
            return None
        img = recreate_aligned_images(image, pts, size)
        img = np.array(img)
        return img



def flip(tensor, is_label=False):
    """Flip an image or a set of heatmaps left-right

    Arguments:
        tensor {numpy.array or torch.tensor} -- [the input image or heatmaps]

    Keyword Arguments:
        is_label {bool} -- [denote wherever the input is an image or a set of heatmaps ] (default: {False})
    """
    if not torch.is_tensor(tensor):
        tensor = torch.from_numpy(tensor)

    if is_label:
        tensor = shuffle_lr(tensor).flip(tensor.ndimension() - 1)
    else:
        tensor = tensor.flip(tensor.ndimension() - 1)

    return tensor

def shuffle_lr(parts, pairs=None):
    """Shuffle the points left-right according to the axis of symmetry
    of the object.

    Arguments:
        parts {torch.tensor} -- a 3D or 4D object containing the
        heatmaps.

    Keyword Arguments:
        pairs {list of integers} -- [order of the flipped points] (default: {None})
    """
    if pairs is None:
        pairs = [16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
                 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 27, 28, 29, 30, 35,
                 34, 33, 32, 31, 45, 44, 43, 42, 47, 46, 39, 38, 37, 36, 41,
                 40, 54, 53, 52, 51, 50, 49, 48, 59, 58, 57, 56, 55, 64, 63,
                 62, 61, 60, 67, 66, 65]
    if parts.ndimension() == 3:
        parts = parts[pairs, ...]
    else:
        parts = parts[:, pairs, ...]

    return parts

if __name__ == '__main__':
    # for test
    ckpt_loc = '/Users/yangjie08/dataset/d_lib_models/shape_predictor_68_face_landmarks.dat'
    image_loc = './jujingwei.jpg'


    fa = FaceAlignment(ckpt_loc)
    image = load_image(image_loc)

    img = fa.rotate_align_crop(image, size=256)
    # print(img.dtype, img.max())  # uint8 RGB
    plt.imshow(img)
    # for detection in pts:  # 几个人
    #     lm_eye_left = detection[36: 42]
    #     lm_eye_right = detection[42: 48]
    # plt.scatter(lm_eye_left[:, 0], lm_eye_left[:, 1], 2)
    plt.show()

