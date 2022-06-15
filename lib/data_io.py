"""Module contains BrainFMRIDataset class to be used in training procedure."""

import scipy.io as iom
import torch, os, sys
from torch.utils.data import Dataset
from torchvision import transforms
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from tools.utils import fmri_masking

IMAGE_SIZE = (53, 63, 52, 490)

def _transformation():
    """Function to implemenat callable transformation.

    Returns:
        callable: transform to be applied on a sample
    """
    return transforms.Compose([
            transforms.Lambda(lambda x: x.unsqueeze(0))
        ])

class BrainFMRIDataset(Dataset):
    """Brain segmentation (parcellation) dataset

    Args:
        images_path (list): list of fMRI files' directories
        components_path (list): list of components' directories - prior
        processing_info (list): list of side information files
        mask_path (str): path to mask file
        valid_components (list): indices of verified components
        component_id (int): target brain network
        min_max_scale (bool, optional): scale input data range [0,1]. Defaults to True.
    """    
    def __init__(self, images_path: list, components_path: list, processing_info: list, mask_path: str, valid_components: list, component_id: int, min_max_scale=True):
        self.images = images_path
        self.ica_maps = components_path
        self.ica_informaion = processing_info
        self.mask_path = mask_path
        self.verified_components = valid_components
        self.ica_map_id = component_id
        self.scaling = min_max_scale
        self.transform = _transformation()

    def _input_volume_load(self, subject_ind: int) -> torch.Tensor:  
        img = fmri_masking(self.images[subject_ind], self.mask_path, nor=True, sc=self.scaling)
        return torch.from_numpy(img.get_fdata()).float()

    def _ica_prior_load(self, subject_ind: int) -> torch.Tensor:    
        param = torch.Tensor(self.verified_components) - 1
        file_content = iom.loadmat(self.ica_maps[subject_ind])
        norm_l1 = torch.abs(torch.einsum('nmk,mjk->njk', torch.Tensor(file_content['ic'])[param.long(),:].T.unsqueeze(1), 
                            torch.Tensor(file_content['tc']) [:,param.long()].unsqueeze(0)))
        norm_l1 = torch.div(norm_l1[:,:,self.ica_map_id], norm_l1.sum(axis=-1)).float()
        file_content = iom.loadmat(self.ica_informaion[subject_ind])        
        vol_indices = torch.Tensor(file_content['sesInfo'][0,0]['mask_ind']).ravel() - 1
        map4d = torch.zeros(torch.prod(torch.Tensor(IMAGE_SIZE[:3])).long(), IMAGE_SIZE[-1], dtype=torch.float32)
        map4d[vol_indices.long(),:] = norm_l1
        return map4d.reshape(*reversed(IMAGE_SIZE[:3]), IMAGE_SIZE[-1]).permute(*reversed(range(len(IMAGE_SIZE[:3]))),3)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self._input_volume_load(idx)
        label = self._ica_prior_load(idx)        
        if self.transform:
            img = self.transform(img)
            label = self.transform(label)
        return img, label

