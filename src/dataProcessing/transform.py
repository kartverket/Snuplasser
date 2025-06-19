import numpy as np
from PIL import Image
import random
import torch
import torchvision.transforms.functional as TF


def resize_and_random_crop(
    img,
    mask,
    target_size,
    img_interpolation_type=Image.BILINEAR,
    mask_interpolation_type=Image.NEAREST,
):
    img_width, img_height = img.size
    target_width, target_height = target_size
    img_aspect_ratio = img_width / img_height
    target_aspect_ratio = target_width / target_height

    # Resize image while preserving aspect ratio
    if img_aspect_ratio > target_aspect_ratio:
        # Width becomes the reference size, but ensure height is at least target_height
        scale_factor = max(target_width / img_width, target_height / img_height)
        new_img_size = (int(img_width * scale_factor), int(img_height * scale_factor))
    else:
        # Height becomes the reference size, but ensure width is at least target_width
        scale_factor = max(target_height / img_height, target_width / img_width)
        new_img_size = (int(img_width * scale_factor), int(img_height * scale_factor))

    img_resized = img.resize(new_img_size, img_interpolation_type)
    mask_resized = mask.resize(new_img_size, mask_interpolation_type)

    # Calculate random coordinates for crop
    x1_max = img_resized.width - target_width
    y1_max = img_resized.height - target_height

    try:
        x1 = random.randint(0, x1_max)
    except Exception as e:
        x1 = 0

    try:
        y1 = random.randint(0, y1_max)
    except Exception as e:
        y1 = 0

    x2 = x1 + target_width
    y2 = y1 + target_height

    # Apply random crop to both image and mask
    img_cropped = img_resized.crop((x1, y1, x2, y2))
    mask_cropped = mask_resized.crop((x1, y1, x2, y2))

    return img_cropped, mask_cropped


class TrainingTransform:
    def __init__(self, image_size, num_classes):
        self.image_size = image_size
        self.num_classes = num_classes
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def __call__(self, img, mask):
        # Resize

        if img.size[0] != self.image_size or img.size[1] != self.image_size:
            img, mask = resize_and_random_crop(img, mask, self.image_size)

        # Random horizontal flip
        if torch.rand(1).item() > 0.4:
            img = TF.hflip(img)
            mask = TF.hflip(mask)

        if torch.rand(1).item() > 0.4:
            img = TF.vflip(img)
            mask = TF.vflip(mask)

        # Other augmentation techniques
        # - Blurring
        # - Color jitter
        # - Random rotation

        # To tensor
        img = TF.to_tensor(img)
        mask = torch.as_tensor(np.array(mask), dtype=torch.int64)

        # Normalize image
        # normalize = transforms.Normalize(mean=self.mean, std=self.std)
        # img = normalize(img)

        return img, mask


def round2nearest_multiple(x, base):
    """
    Round x to the nearest multiple of base.

    Args:
    x (int): Number to be rounded.
    base (int): The base multiple to round to.

    Returns:
    int: Rounded number.
    """
    return base * round(x / base)


def resize_image(img, target_size, interp=Image.BILINEAR):
    """
    Resize the image to the given target size, ensuring the smallest side
    of the image is resized to target size while keeping the aspect ratio.

    Args:
    img (PIL.Image): Input image to resize.
    target_size (int): Desired size for the smallest side of the image.
    imgMaxSize (int): Maximum allowable size for the larger side of the image.
    padding_constant (int): The padding constant to which dimensions should be rounded.

    Returns:
    PIL.Image: Resized image.
    """

    # ori_width, ori_height = img.size

    # Calculate scaling factor
    # scale = min(target_size / float(min(ori_height, ori_width)),
    #            imgMaxSize / float(max(ori_height, ori_width)))

    # target_width = int(ori_width * scale)
    # target_height = int(ori_height * scale)

    # Adjust to be multiples of padding_constant
    # target_width = round2nearest_multiple(target_width, padding_constant)
    # target_height = round2nearest_multiple(target_height, padding_constant)

    # Resize the image
    img_resized = img.resize((target_size, target_size), interp)

    return img_resized


class ValidationTransform:
    def __init__(self, image_size, num_classes):
        self.image_size = image_size
        self.num_classes = num_classes
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def __call__(self, img, mask):
        # Resize
        # img, mask = resize_and_random_crop(img, mask, self.image_size)

        img = resize_image(img, self.image_size[0])
        mask = resize_image(mask, self.image_size[0], interp=Image.NEAREST)

        # To tensor
        img = TF.to_tensor(img)
        mask = torch.as_tensor(np.array(mask), dtype=torch.int64)

        # Normalize image
        # normalize = transforms.Normalize(mean=self.mean, std=self.std)
        # img = normalize(img)

        return img, mask
