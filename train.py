import torch, logging, argparse, os, json, time
import pandas as pd
from torch.optim import lr_scheduler
from lib.data_io import BrainFMRIDataset
from lib.model import BrainSeg

def main():
    parser = argparse.ArgumentParser(description='Training brain segmentation models')
    parser.add_argument('-n', '--network', required=True, help='Network ID') 
    parser.add_argument('-b', '--batchs', required=True, help='Batch size')  
    parser.add_argument('-w', '--workers', required=True, help='Number of workers for data loader')    
    args = parser.parse_args()
    logging.root.setLevel(logging.NOTSET)
    logging.basicConfig(level=logging.NOTSET, format="[ %(asctime)s ]  %(levelname)s : %(message)s", datefmt="%d-%b-%y %H:%M:%S")
    logging.info("Training procedure strated ...")
    if torch.cuda.is_available():
        logging.debug("Available processing unit ({})".format(torch.cuda.get_device_name(0)))
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    valid_networks = [69,53,98,99,45,21,56,3,9,2,11,27,54,66,80,72,16,5,62,15,12,93,20,8,77,
                      68,33,43,70,61,55,63,79,84,96,88,48,81,37,67,38,83,32,40,23,71,17,51,94,13,18,4,7]
    try:
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
        dataloaders = {x: torch.utils.data.DataLoader(data_pack[x], batch_size=int(args.batchs), shuffle=True, num_workers=int(args.workers), pin_memory=True) for x in ['train', 'val']}       
        logging.info("Model, optimizer and criterion configuration")
        segmentation_model = BrainSeg(i_channel=1, h_channel=[64, 32, 16, 8])
        segmentation_model = segmentation_model.to(dev)
        optimizer = torch.optim.Adam(segmentation_model.parameters(), lr=1e-3)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=1)
        criterion = torch.nn.MSELoss(reduction='sum')
        best_loss = float("Inf")
        num_epochs = 200
        phase_error = {}
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
                    optimizer.zero_grad()
                    shuffled_index = torch.randint(inp.shape[-1], (inp.shape[-1],))
                    for j in shuffled_index:
                        with torch.set_grad_enabled(phase == 'train'):
                            preds = segmentation_model(inp[...,j])
                            loss = criterion(preds, label[...,j])
                            if phase == 'train':
                                loss.backward()
                                optimizer.step()
                        running_loss += loss.item()  
                if phase == 'train':
                    scheduler.step()
                epoch_loss = running_loss / len(data_pack[phase])
                phase_error[phase] = epoch_loss
            logging.info("Epoch {}/{} - Train Loss: {:.10f} and Validation Loss: {:.10f}".format(phase_error['train'], phase_error['val']))
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                torch.save({'epoch': epoch,
                            'state_dict': segmentation_model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'loss': phase_error}, 
                            os.path.join(checkpoints_directory, 'network{}'.format(valid_networks[int(args.network)]), 'checkpoint_{}_{}.pth'.format(epoch, time.strftime("%m%d%y_%H%M%S"))))
    except Exception as e:
        logging.error("({}) - {}".format(e.__class__.__name__, e))
    logging.info("Training procedure is done, best checkpoint is saved in the given directory.")

if __name__ == "__main__":
    main()