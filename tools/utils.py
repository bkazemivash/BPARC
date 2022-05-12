"""Tools for processing and visualozing 3D/4D Niimg-like object.

This module contains functions to run different preprocessing / postprocessing steps on input fMRI 
images to feed them to the model or do some statistical analysis."""

import numpy as np
from nilearn.masking import unmask, apply_mask
from nilearn import image
from scipy import ndimage, stats

def image_masking(inp_img: object, mask_img: object) -> object:
    """Function to create a masked Nifti image

    Args:
        inp_img (object): a 3D/4D Niimg-like object
        mask_img (object): a 3D Niimg-like object

    Raises:
        TypeError: If 'inp_img' is not a Niimg-like object
        ValueError: If 'inp_img' is not a 4D object

    Returns:
        object: a 3D/4D Niimg-like masked object
    """
    if not hasattr(inp_img, 'get_fdata'):
        raise TypeError("Input image is not a Nifti file, please check your input!")
    if not (inp_img.ndim in [3, 4]):
        raise ValueError("Shape of input image is not 3D/4D!")
    return unmask(apply_mask(inp_img, mask_img), mask_img)
    



def scale_array(ar: np.ndarray, lb = 0, ub = 1, ax = None) -> np.ndarray:
    """Function to scale input array in range of [lb, ub]

    Args:
        ar (np.ndarray): input array to be scaled
        lb (int, optional): lower bound of scaling function. Defaults to 0.
        ub (int, optional): upper bound of scaling function. Defaults to 1.
        ax (obj:int, optional): axis to apply scaling function. Defaults to None.

    Returns:
        np.ndarray: Scaled array ranging in [lb, ub]
    """
    return lb + ((ub - lb) * (np.subtract(ar, np.min(ar, axis=ax, keepdims=True))) / np.ptp(ar, axis=ax, keepdims=True))
    

def normalize_array(ar: np.ndarray, ax = None) -> np.ndarray:
    """Function to z-score input array

    Args:
        ar (np.ndarray): input array to be normalized
        ax (obj:int, optional): axis to apply scaling function. Defaults to None.

    Returns:
        np.ndarray: normalized (z-score) array
    """
    return stats.zscore(ar, axis=ax)


def apply_norm_scale_fMRI(inp_img: object, mask_img: object, ax = 1, sc = False) -> object:
    """Function to zscore and scale input fMRI image using a mask 

    Args:
        inp_img (object): a 4D Niimg-like object
        mask_img (object): a 3D Niimg-like object
        ax (int, optional): Z-score by a specific axis, {0 for voxel-wise(fMRI), 1 for timepoint-wise(fMRI)}. Defaults to -1.
        sc (bool, optional): If scaling is needed. Defaults to False.

    Raises:
        TypeError: If 'inp_img' is not a Niimg-like object
        ValueError: If 'inp_img' is not a 4D object

    Returns:
        object: a 4D Niimg-like object
    """
    if not hasattr(inp_img, 'get_fdata'):
        raise TypeError("Input image is not a Nifti file, please check your input!")
    if inp_img.ndim != 4:
        raise ValueError("Shape of input fMRI image is not 4D!")
    data = apply_mask(inp_img, mask_img)
    data = normalize_array(data, ax=ax)
    if sc:
        data = scale_array(data, ax=-1)
    return unmask(data, mask_img)


def get_coordinates(inp_img: object, state=False) -> tuple:
    """Function to find real/transformed coordinate of amplitude

    Args:
        inp_img (object): a 3D Niimg-like object
        state (bool, optional): True if image data contains negative values, otherwise False. Defaults to False.

    Raises:
        TypeError: If 'inp_img' is not a Niimg-like object

    Returns:
        tuple: A tuple including real and transformed coordinates of amplitude
    """
    if not hasattr(inp_img, 'get_fdata'):
        raise TypeError("Input image is not a Nifti file, please check your input!")
    data = inp_img.get_fdata()
    assert data.ndim == 3, 'Input image must be a 3D tensor (x, y, z)'
    if state:
        data = np.abs(data)
    key_points = ndimage.maximum_position(data)
    return key_points, image.coord_transform(*key_points, inp_img.affine)

