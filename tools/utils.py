"""Tools for processing and visualozing 3D/4D Niimg-like object.

This module contains mandatory functions to run different preprocessing / postprocessing steps on input fMRI 
images to feed them to the model or do some statistical analysis."""

import numpy as np
from torch import nn
from nilearn.masking import unmask, apply_mask
from nilearn import image
from scipy import ndimage, stats


def weights_init(layer_object: nn.Conv3d or nn.ConvTranspose3d) -> None:
    """Function to initialize model weights using kaiming method

    Args:
        layer_object (nn.Conv3d or nn.ConvTranspose3d): given model's layer
    """
    if isinstance(layer_object, (nn.Conv3d, nn.ConvTranspose3d)):
        nn.init.kaiming_uniform_(layer_object.weight, nonlinearity='relu')
        nn.init.zeros_(layer_object.bias)
   

def scale_array(ar: np.ndarray, lb = 0, ub = 1, ax = None) -> np.ndarray:
    """Function to scale input array in range of [lb, ub]

    Args:
        ar (np.ndarray): input array to be scaled
        lb (int, optional): lower bound of scaling function. Defaults to 0.
        ub (int, optional): upper bound of scaling function. Defaults to 1.        
        ax (obj:int, optional): axis to apply scaling function. Defaults to None.

    Returns:
        np.ndarray: scaled array ranging in [lb, ub]
    """
    return lb + ((ub - lb) * (np.subtract(ar, np.min(ar, axis=ax, keepdims=True))) / np.ptp(ar, axis=ax, keepdims=True))
    

def normalize_array(ar: np.ndarray, ax = None) -> np.ndarray:
    """Function to z-score input array

    Args:
        ar (np.ndarray): input array to be normalized.         
        ax (obj:int, optional): axis to apply scaling function. Defaults to None.

    Returns:
        np.ndarray: normalized (z-score) array
    """
    return stats.zscore(ar, axis=ax)


def fmri_masking(inp_img: str, mask_img: str, ax = 1, nor = False, sc = False) -> object:
    """Function to zscore and scale input fMRI image using a mask 

    Args:
        inp_img (str): path to a 4D Niimg-like object
        mask_img (str): path to a 3D Niimg-like object
        ax (int, optional): z-score by a specific axis; 0 for voxel-wise(fMRI), 1 for timepoint-wise(fMRI). Defaults to -1.
        nor (bool, optional): True if normalization is needed. Defaults to False.
        sc (bool, optional): True if scaling is needed. Defaults to False.

    Raises:
        TypeError: if 'inp_img' or 'mask_img' is not a Niimg-like object

    Returns:
        object: a 4D Niimg-like object
    """
    if not (inp_img.lower().endswith(('.nii', '.nii.gz')) and mask_img.lower().endswith(('.nii', '.nii.gz'))):
        raise TypeError("Input image/mask is not a Nifti file, please check your input!")
    data = apply_mask(inp_img, mask_img)
    if nor:
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
        TypeError: if 'inp_img' is not a Niimg-like object

    Returns:
        tuple: a tuple including real and transformed coordinates of amplitude
    """
    if not hasattr(inp_img, 'get_fdata'):
        raise TypeError("Input image is not a Nifti file, please check your input!")
    data = inp_img.get_fdata()
    assert data.ndim == 3, 'Input image must be a 3D tensor (x, y, z)'
    if state:
        data = np.abs(data)
    key_points = ndimage.maximum_position(data)
    return key_points, image.coord_transform(*key_points, inp_img.affine)

