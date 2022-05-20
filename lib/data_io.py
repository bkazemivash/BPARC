"""Module contains BrainFMRIDataset class to be used in training procedure."""

import scipy.io as iom
import torch, os, sys
from torch.utils.data import Dataset
from torchvision import transforms
from nilearn.masking import unmask, apply_mask
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from tools.utils import normalize_array, scale_array

IMAGE_SIZE = (53, 63, 52, 490)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def _convert_index(ind):
    """Function to convert a given index to a specific fMRI file and a volume

    Args:
        ind (int): given index of a volume in dataset.

    Returns:
        tuple: subject index, volume index
    """
    return ind // IMAGE_SIZE[-1], ind % IMAGE_SIZE[-1]

def _transformation():
    """Function to implemenat callable transformation.

    Returns:
        callable: transform to be applied on a sample
    """
    return transforms.Compose([
            transforms.Lambda(lambda x: x.to(DEVICE)),
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

    def _input_volume_load(self, subject_ind: int, volume_ind: int) -> torch.Tensor:  
        img = apply_mask(self.images[subject_ind], self.mask_path)[volume_ind,:]
        img = normalize_array(img)
        if self.scaling:
            img = scale_array(img) 
        return torch.from_numpy(unmask(img, self.mask_path).get_fdata()).float()

    def _ica_prior_load(self, subject_ind: int, volume_ind: int) -> torch.Tensor:    
        param = torch.Tensor(self.verified_components) - 1
        file_content = iom.loadmat(self.ica_maps[subject_ind])
        spatial_map = torch.unsqueeze(torch.Tensor(file_content['ic']).T [...,param.long()], 0)
        time_coarse = torch.unsqueeze(torch.Tensor(file_content['tc']) [...,param.long()],1)
        norm_l1 = torch.abs(torch.einsum('nmk,mjk->njk',time_coarse, spatial_map)).sum(axis=-1)
        p_map = torch.div(torch.multiply(time_coarse[volume_ind,0, self.ica_map_id], spatial_map[0,:,self.ica_map_id]), norm_l1[volume_ind,:]).abs().float()
        file_content = iom.loadmat(self.ica_informaion[subject_ind])
        vol_indices = torch.Tensor(file_content['sesInfo'][0,0]['mask_ind']).ravel() - 1
        temp_volume = torch.zeros(torch.prod(torch.Tensor(IMAGE_SIZE[:3])).long(), dtype=torch.float32)
        temp_volume[vol_indices.long()] = p_map
        return temp_volume.reshape(*reversed(IMAGE_SIZE[:3])).permute(*reversed(range(len(IMAGE_SIZE[:3]))))

    def __len__(self):
        return len(self.images) * IMAGE_SIZE[-1]

    def __getitem__(self, idx):
        subject_id, volume_id = _convert_index(idx)
        img = self._input_volume_load(subject_id, volume_id)
        label = self._ica_prior_load(subject_id, volume_id)        
        if self.transform:
            img = self.transform(img)
            label = self.transform(label)
        sample = {'image': img, 'prior': label}
        return sample

