"""Functions for processing and visualization.

This module contains mandatory functions to run different preprocessing / postprocessing steps on input fMRI 
images."""

import numpy as np
import operator
import torch.nn as nn
from typing import Tuple, Union
from nilearn.masking import unmask, apply_mask
from nibabel.nifti1 import Nifti1Image
from nilearn import image
from scipy import ndimage, stats
from functools import reduce


def weights_init(m: nn.Module) -> None:
    """Function to initialize model weights using kaiming method

    Args:
        m (nn.Module): model layers
    """
    if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
        nn.init.kaiming_uniform_(m.weight.data,nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        nn.init.constant_(m.weight.data, 1)
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight.data)
        nn.init.constant_(m.bias.data, 0)     
   
def tuple_prod(inp: Tuple[int]):
    """Function multiplies all elements of the input tuple.
       -- Equivalent to numpy.prod, but with lower running time in our case
    Args:
        inp (Tuple[int]): input tuple

    Returns:
        int: multiplication of all elements
    """    
    return reduce(operator.mul, inp, 1)

def scale_array(ar: np.ndarray, lb = 0, ub = 1, ax = -1) -> np.ndarray:
    """Function to scale input array in range of [lb, ub]

    Args:
        ar (np.ndarray): input array to be scaled
        lb (int, optional): lower bound of scaling function. Defaults to 0.
        ub (int, optional): upper bound of scaling function. Defaults to 1.        
        ax (int, optional): axis to apply scaling function. Defaults to -1.

    Returns:
        np.ndarray: scaled array ranging in [lb, ub]
    """
    return lb + ((ub - lb) * (np.subtract(ar, np.min(ar, axis=ax, keepdims=True))) / np.ptp(ar, axis=ax, keepdims=True))
    

def normalize_array(ar: np.ndarray, ax = -1) -> np.ndarray:
    """Function to z-score input array

    Args:
        ar (np.ndarray): input array to be normalized.         
        ax (obj:int, optional): axis to apply scaling function. Defaults to None.

    Returns:
        np.ndarray: normalized (z-score) array
    """
    return stats.zscore(ar, axis=ax)


def fmri_masking(inp_img: Union[str, Nifti1Image], mask_img: str, ax = 1, nor = False, sc = False) -> object:
    """Function to z-score and scale input fMRI image using a mask 

    Args:
        inp_img (Union[str, Nifti1Image]): a 4D Niimg-like object or a path to it
        mask_img (str): path to a 3D Niimg-like object
        ax (int, optional): z-score by a specific axis; 0 for voxel-wise(fMRI), 1 for timepoint-wise(fMRI). Defaults to 1.
        nor (bool, optional): True if normalization is needed. Defaults to False.
        sc (bool, optional): True if scaling is needed. Defaults to False.

    Raises:
        TypeError: if 'inp_img' or 'mask_img' is not a Niimg-like object

    Returns:
        object: a 4D Niimg-like object
    """    
    if (isinstance(inp_img, str) and not (inp_img.lower().endswith(('.nii', '.nii.gz')))):
        if not isinstance(inp_img, Nifti1Image):
            raise TypeError("Input image/mask is not a Nifti file, please check your input!") 
    data = apply_mask(inp_img, mask_img)
    if nor:
        data = normalize_array(data, ax=ax)
    if sc:
        data = scale_array(data, ax=-1)
    return unmask(data, mask_img)


def get_coordinates(inp_img: Nifti1Image, state=False) -> Tuple:
    """Function to find original/MNI coordinate of amplitude

    Args:
        inp_img (Nifti1Image): a 3D Niimg-like object
        state (bool, optional): True if image data contains negative values, otherwise False. Defaults to False.

    Raises:
        TypeError: if 'inp_img' is not a Niimg-like object

    Returns:
        Tuple: a tuple including real and MNI coordinates of amplitude
    """
    if not hasattr(inp_img, 'get_fdata'):
        raise TypeError("Input image is not a Nifti1Image file, please check your input!")
    data = inp_img.get_fdata()
    assert data.ndim == 3, 'Input image must be a 3D tensor (x, y, z)'
    if state:
        data = np.abs(data)
    key_points = ndimage.maximum_position(data)
    return key_points, image.coord_transform(*key_points, inp_img.affine) # type: ignore
