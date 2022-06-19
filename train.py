import torch, logging, argparse, os, json, time
import pandas as pd
from torch.optim import lr_scheduler
from torch.autograd import Variable
from lib.data_io import BrainFMRIDataset
from lib.model import BrainSeg, BrainSegPP
from tools.utils import weights_init

def main():
    parser = argparse.ArgumentParser(description='Training brain segmentation models')
    parser.add_argument('-n', '--network', required=True, help='Network ID') 
    parser.add_argument('-b', '--num_batches', required=True, help='Batch size')  
    parser.add_argument('-w', '--num_workers', required=True, help='Number of workers for data loader')    
    parser.add_argument('-e', '--num_epochs', required=True, help='Number of epochs for training phase')  
    parser.add_argument('-s', '--status', required=True, help='Model configuration: BPARC vs BPARC++')   
    parser.add_argument('-l', '--loss_fun', required=False, default='MSE', help='Loss function for training the model')
    parser.add_argument('-r', '--learning_rate', required=False, default=1e-3, help='Learning rate for training the model')
    parser.add_argument('-d', '--decay_rate', required=False, default=.1, help='Learning decay for training the model') 
    parser.add_argument('-v', '--save_model', required=False, default=True, help='Flag for saving the model') 
    args = parser.parse_args()
    logging.root.setLevel(logging.NOTSET)
    logging.basicConfig(level=logging.NOTSET, format="[ %(asctime)s ]  %(levelname)s : %(message)s", datefmt="%d-%b-%y %H:%M:%S")
    for i in range(torch.cuda.device_count()):
        logging.debug("Available processing unit ({} : {})".format(i, torch.cuda.get_device_name(i)))
    BPARC_PLUS_PLUS = args.status == "True"
    SAVE_FLAG = args.save_model == "True"
    logging.info("Training procedure strated with {}".format("BPARC++" if BPARC_PLUS_PLUS else "BPARC"))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    valid_networks = [69,53,98,99,45,21,56,3,9,2,11,27,54,66,80,72,16,5,62,15,12,93,20,8,77,
                      68,33,43,70,61,55,63,79,84,96,88,48,81,37,67,38,83,32,40,23,71,17,51,94,13,18,4,7]
    loss_function = args.loss_fun
    if not (loss_function in ['MSE', 'KLD', 'COS']):
        raise ValueError("Loss function is not valid, only MSE, KLD, and COS are accepted!")   
    setting_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)) , "setting.json")
    if not (os.path.exists(setting_file_path)):
        raise FileNotFoundError("Setting file not found")  
    with open(setting_file_path) as file:
        conf = json.load(file)
    table_file_path = os.path.abspath(conf['train_data']['file_path'])
    checkpoints_directory = os.path.abspath(conf['check_points']['file_path'])
    mask_file_path = os.path.abspath(conf['mask']['file_path'])
    if not (os.path.exists(table_file_path)):
        raise FileNotFoundError("Table of input fMRI files not found, check setting.json file!") 
    if not (os.path.exists(checkpoints_directory)):
        raise FileNotFoundError("Trained models' saving directory not found, check setting.json file!")  
    if not (os.path.exists(mask_file_path)):
        raise FileNotFoundError("Mask image file not found, check setting.json file!")       
    train_data = pd.read_pickle(table_file_path)[:100]  
    logging.info("Loading subjects fMRI files and component maps")
    main_dataset = BrainFMRIDataset(train_data['fMRI'].tolist(),train_data['components'].tolist(),
                                    train_data['side'].tolist(), mask_file_path, valid_networks,
                                    int(args.network))
    data_pack = {}
    data_pack['train'], data_pack['val'] = torch.utils.data.random_split(main_dataset, [80, 20])
    dataloaders = {x: torch.utils.data.DataLoader(data_pack[x], batch_size=int(args.num_batches), shuffle=True, num_workers=int(args.num_workers), pin_memory=True) for x in ['train', 'val']}       
    gpu_ids = list(range(torch.cuda.device_count()))
    if BPARC_PLUS_PLUS:
        segmentation_model = BrainSegPP(i_channel=1, h_channel=[64, 32, 16, 8])
    else:
        segmentation_model = BrainSeg(i_channel=1, h_channel=[64, 32, 16, 8])
    segmentation_model.apply(weights_init)
    if torch.cuda.device_count() > 1:
        segmentation_model = torch.nn.DataParallel(segmentation_model, device_ids = gpu_ids)
        logging.info("Pytorch data Parallel activated.")
    segmentation_model = segmentation_model.cuda()
    optimizer = torch.optim.Adam(segmentation_model.parameters(), lr=float(args.learning_rate))
    scheduler = lr_scheduler.StepLR(optimizer, step_size=40, gamma=float(args.decay_rate))
    if loss_function == 'MSE':
        criterion = torch.nn.MSELoss(reduction='sum')
    elif loss_function == 'KLD':
        criterion = torch.nn.KLDivLoss(reduction='sum')
    else:
        criterion = torch.nn.CosineEmbeddingLoss(reduction='sum')
    best_loss = float("inf")
    logging.info(f"Optimizer: Adam , Criterion: {loss_function} , lr: {args.learning_rate} , decay: {args.decay_rate}")
    num_epochs = int(args.num_epochs)
    phase_error = {'train': float("inf"), 'val': float("inf")}
    logging.info("Start training procedure, model is running on GPU : {}".format(next(segmentation_model.parameters()).is_cuda))
    for epoch in range(num_epochs):
        for phase in ['train', 'val']:
            if phase == 'train':
                segmentation_model.train() 
            else:
                segmentation_model.eval() 
            running_loss = 0.0
            for inp, label in dataloaders[phase]:
                inp = inp.to(dev, non_blocking=True)
                label = label.to(dev, non_blocking=True)
                shuffled_index = torch.randint(inp.shape[-1], (inp.shape[-1],))
                with torch.set_grad_enabled(phase == 'train'):
                    for j in shuffled_index:
                        optimizer.zero_grad()
                        preds = segmentation_model(inp[...,j])
                        masker = label[...,j].gt(0.0)
                        if loss_function == 'MSE':
                            loss = criterion(torch.masked_select(preds, masker), torch.masked_select(label[...,j], masker))      
                        elif loss_function == 'KLD': 
                            loss = criterion(torch.nn.LogSoftmax(dim=-1)(torch.masked_select(preds, masker)), 
                                            torch.nn.Softmax(dim=-1)(torch.masked_select(label[...,j], masker)))
                        else:
                            loss = criterion(torch.masked_select(preds, masker).unsqueeze(0), torch.masked_select(label[...,j], masker).unsqueeze(0), 
                                            Variable(torch.ones(1)).cuda()) 
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
                        running_loss += loss.item()                     
            if phase == 'train':
                scheduler.step()
            epoch_loss = running_loss / (len(data_pack[phase]) * inp.shape[-1])
            phase_error[phase] = epoch_loss
        logging.info("Epoch {}/{} - Train Loss: {:.10f} and Validation Loss: {:.10f}".format(epoch+1, num_epochs, phase_error['train'], phase_error['val']))
        if phase == 'val' and epoch_loss <= best_loss and SAVE_FLAG:
            best_loss = epoch_loss
            torch.save({'epoch': epoch,
                        'state_dict': segmentation_model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'loss': phase_error}, 
                        os.path.join(checkpoints_directory, 'network{}'.format(valid_networks[int(args.network)]), 'checkpoint_{}_{}.pth'.format(epoch, time.strftime("%m%d%y_%H%M%S"))))


if __name__ == "__main__":
    main()