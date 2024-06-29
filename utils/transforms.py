from __future__ import division
import random
from PIL import Image
import torchvision.transforms.functional as F
from torchvision.transforms.transforms import Compose, Lambda
import numpy as np
import numbers
import torch

class ToTensor(object):

    def __call__(self, images):
        tensor = []
        for image in images:
            tensor.append(F.to_tensor(image))
        return tensor
        # return torch.stack(tensor, dim=0)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class Normalize(object):
    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace
    def __call__(self, tensor):
        batch_size = len(tensor)
        res = []
        for i in range(0, batch_size):
            res.append(F.normalize(tensor[i], self.mean, self.std, self.inplace))
        # return torch.stack(res, dim=0)
        return res

class Grayscale(object):
    """Convert image to grayscale.

    Args:
        num_output_channels (int): (1 or 3) number of channels desired for output image

    Returns:
        PIL Image: Grayscale version of the input.
        - If num_output_channels == 1 : returned image is single channel
        - If num_output_channels == 3 : returned image is 3 channel with r == g == b

    """

    def __init__(self, num_output_channels=1):
        self.num_output_channels = num_output_channels

    def __call__(self, imgs):
        """
        Args:
            img (PIL Image): Image to be converted to grayscale.

        Returns:
            PIL Image: Randomly grayscaled image.
        """
        result = []
        for img in imgs:
            img = Image.fromarray(img)
            grayed = F.to_grayscale(img, num_output_channels=self.num_output_channels)
            result.append(np.asarray(grayed))

        return result

class Resize(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        assert isinstance(size, int) or (isinstance(size, Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, iamges):
        """
        Args:
            img (PIL Image): Image to be scaled.

        Returns:
            PIL Image: Rescaled image.
        """
        res = []
        for image in iamges:
            res.append(F.resize(image, self.size, self.interpolation))
        return res


class RandomCrop(object):
    def __init__(self, size, padding=None, pad_if_needed=False, fill=0, padding_mode='constant'):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.fill = fill
        self.padding_mode = padding_mode

    @staticmethod
    def get_params(img, output_size):
        """Get parameters for ``crop`` for a random crop.

        Args:
            img (PIL Image): Image to be cropped.
            output_size (tuple): Expected output size of the crop.

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        w, h = _get_image_size(img)
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, imgs):
        """
        Args:
            img (PIL Image): Image to be cropped.

        Returns:
            PIL Image: Cropped image.
        """
        res = []
        for image in imgs:
            if self.padding is not None:
                image = F.pad(image, self.padding, self.fill, self.padding_mode)

            # pad the width if needed
            if self.pad_if_needed and image.size[0] < self.size[1]:
                image = F.pad(image, (self.size[1] - image.size[0], 0), self.fill, self.padding_mode)
            # pad the height if needed
            if self.pad_if_needed and image.size[1] < self.size[0]:
                image = F.pad(image, (0, self.size[0] - image.size[1]), self.fill, self.padding_mode)

            i, j, h, w = self.get_params(image, self.size)
            res.append(F.crop(image, i, j, h, w))
        return res

    def __repr__(self):
        return self.__class__.__name__ + '(size={0}, padding={1})'.format(self.size, self.padding)


def _get_image_size(img):
    if F._is_pil_image(img):
        return img.size
    elif isinstance(img, torch.Tensor) and img.dim() > 2:
        return img.shape[-2:][::-1]
    else:
        raise TypeError("Unexpected type {}".format(type(img)))

class RandomHorizontalFlip(torch.nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, images):
        """
        Args:
            img (PIL Image or Tensor): Image to be flipped.

        Returns:
            PIL Image or Tensor: Randomly flipped image.
        """
        res = []
        flag = torch.rand(1) < self.p
        for image in images:
            if flag:
                res.append(F.hflip(image))
            else:
                res.append(image)
        return res


    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


# class MyRandomCrop(object):
#     def __init__(self, size):
#         if isinstance(size, numbers.Number):
#             self.size = (int(size), int(size))
#         else:
#             self.size = size
#
#     @staticmethod
#     def get_params(img, outsize):
#         depth, h, w = img.shape
#         th, tw = outsize
#
#         if w == tw and h == th:
#             return 0, 0, th, tw
#
#         i = random.randint(0, h - th)
#         j = random.randint(0, w - tw)
#
#         return i, j, th, tw
#
#     def __call__(self, imgs):
#         """
#         :param imgs: seq_len, image_channel, image_height, image_width
#         :return:
#         """
#         cropped_imgs = []
#         for img in imgs:
#             i, j, h, w = self.get_params(img, self.size)
#             cropped_imgs.append(img[:, i:i + h, j:j + w])
#
#         return cropped_imgs


class CenterCrop(object):
    """Crops the given PIL Image at the center.

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    @staticmethod
    def get_params(img, output_size):
        depth, h, w = img.shape
        th, tw = output_size
        i = int(round((h - th) / 2.))
        j = int(round((w - tw) / 2.))
        return i, j, th, tw

    def __call__(self, imgs):
        """
        Args:
            imgs: seq_len, image_channel, image_height, image_width

        Returns:
            Cropped image.
        """
        cropped_imgs = []
        for img in imgs:
            i, j, h, w = self.get_params(img, self.size)
            cropped_imgs.append(img[:, i:i + h, j:j + w])
        return cropped_imgs

class TemporalRescale(object):
    def __init__(self, temp_scaling=0.2):
        self.min_len = 32
        self.max_len = 230
        self.L = 1.0 - temp_scaling
        self.U = 1.0 + temp_scaling

    def __call__(self, clip):
        vid_len = len(clip)
        new_len = int(vid_len * (self.L + (self.U - self.L) * np.random.random()))
        if new_len < self.min_len:
            new_len = self.min_len
        if new_len > self.max_len:
            new_len = self.max_len
        if (new_len - 4) % 4 != 0:
            new_len += 4 - (new_len - 4) % 4
        if new_len <= vid_len:
            index = sorted(random.sample(range(vid_len), new_len))
        else:
            index = sorted(random.choices(range(vid_len), k=new_len))
        return clip[index]