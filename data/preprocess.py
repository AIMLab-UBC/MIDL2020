from PIL import Image
import numpy as np
import torchvision
import torch

COLOR_JITTER = torchvision.transforms.ColorJitter(64.0 / 255, 0.75, 0.25, 0.04)


def check_luminance(np_image, blank_thresh=210, blank_percent=0.75):
    """Function to check the patch is background or not

    Parameters
    ----------
    np_image : numpy array
        A pillow image contains a pillow image

    is_eval : bool
        If true, it returns a 4D tensor
        If false, it returns a 3D tensor

    Returns
    -------
        Return true if it is not background
    """
    image_luminance = 0.2126 * np_image[:, :, 0] + \
        0.7152 * np_image[:, :, 1] + 0.0722 * np_image[:, :, 2]
    return np.mean(image_luminance > blank_thresh) < blank_percent


def raw(image, is_eval=False, apply_color_jitter=True):
    """Function to preprocess images without other preprocessing techniques, such as stain normalization

    Parameters
    ----------
    image : Pillow image
        A pillow image contains a pillow image

    is_eval : bool
        If true, it returns a 4D tensor
        If false, it returns a 3D tensor

    apply_color_jitter: bool
        Apply color jitter only during training

    Returns
    -------
    tensor_image:
        A tensor contains PyTorch image
    """

    np_image = np.asarray(COLOR_JITTER(image)).copy() if apply_color_jitter else np.asarray(image).copy()
    np_image = (np_image - 128.) / 128.
    image_tensor = image_numpy_to_tensor(np_image, is_eval)
    return image_tensor


def image_numpy_to_tensor(np_image, is_eval):
    """Function to convert image numpy array to PyTorch tensor

    Parameters
    ----------
    np_image : numpy array
        An numpy array contains a pillow image

    is_eval : bool
        If true, it returns a 4D tensor
        If false, it returns a 3D tensor

    Returns
    -------
    tensor_image:
        A tensor contains PyTorch image
    """
    np_image = np_image.copy().transpose(2, 0, 1)
    tensor_image = torch.from_numpy(np_image).type(torch.float).cuda()
    tensor_image = tensor_image.reshape(
        1, *tensor_image.shape) if is_eval else tensor_image
    return tensor_image


def extract_and_resize(slide, location_width, location_height, extract_size, resize_size):
    """Function to extract a patch from slide at (location_width, location_height) and then resize
        using Lanczos resampling filter

    Parameters
    ----------
    slide : OpenSlide object
        An numpy array contains a pillow image

    location_width : int
        Patch location width

    location_height : int
        Patch location height

    extract_size : int
        Extract patch size

    resize_size : int
        Resize patch size

    Returns
    -------
    patch : Pillow image
        A resized patch
    """
    patch = slide.read_region(
        (location_width, location_height), 0, (extract_size, extract_size)).convert('RGB')
    if extract_size != resize_size:
        patch = patch.resize((resize_size, resize_size),
                             resample=Image.LANCZOS)
    return patch
