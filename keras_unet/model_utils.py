
import numpy as np
from skimage.exposure import rescale_intensity
import random
from skimage.transform import rotate, AffineTransform, warp


def normalize_MRIvolume(MRI_volume):
    """
    Normalize the intensity of the MRI volume.
    """
    p10 = np.percentile(MRI_volume, 10)
    p99 = np.percentile(MRI_volume, 99)
    MRI_volume = rescale_intensity(MRI_volume, in_range=(p10, p99))
    m = np.mean(MRI_volume, axis=(0, 1, 2))
    s = np.std(MRI_volume, axis=(0, 1, 2))
    MRI_volume = (MRI_volume - m) / s
    return MRI_volume

def rotate_clockwise(image):
    rot_ang = random.randint(0,100)
    return rotate(image,-rot_ang)

def rotate_counterclockwise(image):
    rot_ang = random.randint(0,100)
    return rotate(image,rot_ang)

def flip_image(image):
    return np.fliplr(image)


