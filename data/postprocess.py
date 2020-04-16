from PIL import Image
import numpy as np


def hori_concat_img(images):
    """Function to concatenate images horizontally for display purposes

    Parameters
    ----------
    images : list
        List of pillow images

    Returns
    -------
    new_im : numpy array
        Concatenated images
    """
    C = 1
    if len(images.shape) == 4:
        B, C, H, W = images.shape
    else:
        B, H, W = images.shape

    mode = 'L' if C == 1 else 'RGB'

    total_width = W * B
    max_height = H

    new_im = Image.new(mode, (total_width, max_height))

    x_offset = 0
    for i in range(B):
        if C == 3:
            im = Image.fromarray(images[i].transpose(1, 2, 0))
        else:
            im = Image.fromarray(images[i])
        new_im.paste(im, (x_offset, 0))
        x_offset += im.size[0]

    return np.asarray(new_im)
