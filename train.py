import torch, logging, argparse, os, time
import pandas as pd

from torch.optim import lr_scheduler
from torch.utils.data import random_split, DataLoader
from torch.nn.parallel import DataParallel
from torch.nn import CosineSimilarity
from lib.data_io import BrainFMRIDataset
from lib.bparc_unet import BaseUnetModel
from tools.utils import weights_init
from omegaconf import OmegaConf

def criterion(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    cos = CosineSimilarity(dim=1, eps=1e-6)
    pearson = cos(x1 - x1.mean(dim=1, keepdim=True), x2 - x2.mean(dim=1, keepdim=True))
    return 1. - pearson

def main():
    parser = argparse.ArgumentParser(description='Training dense prediction model')
    parser.add_argument('-c', '--config', required=True, help='Path to the config file') 
    parser.add_argument('-m', '--mask', required=True, help='Path to the mask file')  
    parser.add_argument('-t', '--train_set', required=True, help='Path to pandas dataframe that keeps list of images')    
    parser.add_argument('-s', '--save_dir', required=True, help='Path to save checkpoint')  
    args = parser.parse_args()
    
    logging.root.setLevel(logging.NOTSET)
    logging.basicConfig(level=logging.NOTSET, format="[ %(asctime)s ]  %(levelname)s : %(message)s", datefmt="%d-%b-%y %H:%M:%S")
    for i in range(torch.cuda.device_count()):
        logging.debug("Available processing unit ({} : {})".format(i, torch.cuda.get_device_name(i)))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not (os.path.exists(args.mask)):
        raise FileNotFoundError(f"Mask file not found: {args.mask}")       
    if not (os.path.exists(args.config)):
        raise FileNotFoundError(f"Config file not found: {args.config}")  
    if not (os.path.exists(args.train_set)):
        raise FileNotFoundError(f"DataTable: file not found: {args.train_set}") 
    if not (os.path.exists(args.save_dir)):
        raise FileNotFoundError(f"Save directory does not exist, {args.save_dir}")  
    
    logging.info("Loading configuration data ...")
    conf = OmegaConf.load(args.config)
    save_flag = bool(conf.EXPERIMENT.SAVE_MODEL)
    valid_networks = tuple(conf.DATASET.VALID_NETWORKS)
    checkpoints_directory = os.path.abspath(args.save_dir)
    mask_file_path = os.path.abspath(args.mask)
    sample_size = int(conf.DATASET.SAMPLE_SIZE)
    sample_shape = tuple(conf.DATASET.IMAGE_SIZE)
    network_index = int(conf.EXPERIMENT.NETWORK_ID) - 1
    train_data = pd.read_pickle(args.train_set)[:sample_size]
    logging.info("Loading subjects fMRI files and component maps")
    main_dataset = BrainFMRIDataset(train_data['fMRI'].tolist(),train_data['components'].tolist(),
                                    train_data['side'].tolist(), mask_file_path, valid_networks,
                                    network_index, sample_shape, True)
    data_pack = {}
    data_pack['train'], data_pack['val'] = random_split(main_dataset, [80, 20], generator=torch.Generator().manual_seed(70))
    dataloaders = {x: DataLoader(data_pack[x], batch_size=int(conf.TRAIN.BATCH_SIZE), shuffle=True, num_workers=int(conf.TRAIN.WORKERS), pin_memory=True) for x in ['train', 'val']}       
    gpu_ids = list(range(torch.cuda.device_count()))
    base_model = BaseUnetModel(kernel=int(conf.MODEL.KERNEL_SIZE), 
                                use_drop=bool(conf.MODEL.DROP_OUT), 
                                drop_ratio=float(conf.MODEL.DROP_RATE))
    base_model.apply(weights_init)
    if torch.cuda.device_count() > 1:
        base_model = DataParallel(base_model, device_ids = gpu_ids)
        logging.info("Pytorch Distributed Data Parallel activated using gpus: {gpu_ids}")
    if torch.cuda.is_available():
        base_model = base_model.cuda()
    optimizer = torch.optim.Adam(base_model.parameters(), lr=float(conf.TRAIN.BASE_LR))
    scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=float(conf.TRAIN.WEIGHT_DECAY))    
    best_loss = 2.
    logging.info(f"Optimizer: Adam , Criterion: CosineSimilarity , lr: {conf.TRAIN.BASE_LR} , decay: {conf.TRAIN.WEIGHT_DECAY}")
    num_epochs = int(conf.TRAIN.EPOCHS)
    phase_error = {'train': float("-inf"), 'val': float("-inf")}    
    for epoch in range(num_epochs):
        for phase in ['train', 'val']:
            if phase == 'train':
                base_model.train() 
            else:
                base_model.eval() 
            running_loss = 0.0
            for inp, label in dataloaders[phase]:
                inp = inp.to(dev, non_blocking=True)
                label = label.to(dev, non_blocking=True)
                shuffled_index = torch.randint(inp.shape[-1], (inp.shape[-1],))
                with torch.set_grad_enabled(phase == 'train'):
                    for j in shuffled_index:
                        optimizer.zero_grad()
                        preds = base_model(inp[...,j])
                        masker = label[...,j].gt(0.0)
                        loss = criterion(torch.masked_select(preds, masker).unsqueeze(0), torch.masked_select(label[...,j], masker).unsqueeze(0))
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
                        running_loss += loss.item()                     
            if phase == 'train':
                scheduler.step()
            epoch_loss = running_loss / (len(data_pack[phase]) * inp.shape[-1])
            phase_error[phase] = epoch_loss
        logging.info("Epoch {}/{} - Train Loss: {:.10f} and Validation Loss: {:.10f}".format(epoch+1, num_epochs, phase_error['train'], phase_error['val']))
        if phase == 'val' and epoch_loss < best_loss and save_flag:
            best_loss = epoch_loss
            torch.save({'epoch': epoch,
                        'state_dict': base_model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'loss': phase_error,
                        'network': conf.EXPERIMENT.NETWORK_ID}, 
                        os.path.join(checkpoints_directory, 'checkpoint_{}_{}.pth'.format(epoch, time.strftime("%m%d%y_%H%M%S"))))


if __name__ == "__main__":
    main()