"""Module for predicting brain maps

This module contains an implementation for model prediction function to generate brain maps using pretrained
models. You can save mandatory information in .json file before using this module like a path to the .pkl file 
including all models' information, a path to a mask file, and also storage directory. There are 2 command line arguments which define 
brain network domain ('ALL', 'SC', 'AU', 'SM', 'VI', 'CC', 'DM', 'CB') and a path to Nifti image or a DataFrame table
of all subjects.

Example:
    You can easily use this module by calling :
        $ python -W ignore  predict.py  -d AU -i specific_directory/img1.nii 

This source code is licensed under the instructions found in the LICENSE file in the root directory of this source tree.
"""


import torch, os, argparse, logging, time, json
import pandas as pd
from nilearn.image import load_img, new_img_like
from tools.utils import apply_norm_scale_fMRI, image_masking
from lib.model import BrainSeg


def load_model(model_path: str) -> BrainSeg:
    """Function to load a pretrained model

    Args:
        model_path (str): path to saved pretrained model

    Returns:
        BrainSeg: A pretrained model in eval mode
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(os.path.abspath(model_path), map_location=device)
    model = BrainSeg(i_channel=1, h_channel=[64, 32, 16, 8]).to(device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval() 
    return model


def predict_map(model: BrainSeg, inp: str, mask_dir: str, map_id: str, saving_adr: str) -> list:
    """Function to predict brain maps using the pretrained model

    Args:
        model (BrainSeg): A pretrained model in eval mode
        inp (str): Path to a Nifti image /or an information table (DataFrame) of subjects
        mask_dir (str): Path to a 3D Nifit like object
        map_id (str): ID of the model - brain network
        saving_adr (str): Path to storage directory

    Returns:
        list: list of saved maps
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    mask_data = load_img(mask_dir)
    log_list = []
    if inp.lower().endswith('.nii'):
        table = pd.DataFrame([['NoData', inp, os.path.basename(inp), 0, 'NoData', 0, 'NoData', 0]], columns= ['TableName', 'Address', 'SubjectID', 'Age', 'Gender', 'Site', 'Diagnosis', 'Description'])
    else:
        table = pd.read_pickle(os.path.abspath(inp))
    logging.info("Input subject table with size {} is loaded!".format(len(table)))
    for _,row in table.iterrows():
        img_data = load_img(row['Address'])        
        img_data = apply_norm_scale_fMRI(img_data, mask_data, sc=True).get_fdata()
        img_data = torch.from_numpy(img_data).to(device, dtype=torch.float)
        out_img = torch.zeros(img_data.shape)
        with torch.no_grad():
            out_img = [torch.squeeze(model(torch.unsqueeze(torch.unsqueeze(img_data[...,i], dim=0), dim=0))) for i in range(img_data.shape[-1])]
        out_img = torch.stack(out_img, dim=-1)
        processed_img = new_img_like(mask_data, out_img.cpu().data.numpy())
        processed_img = image_masking(processed_img, mask_data)
        save_path = os.path.join(saving_adr, map_id, row['SubjectID'])
        processed_img.to_filename(save_path)   
        log_list.append(save_path)
    return log_list 


def main():
    logging.root.setLevel(logging.NOTSET)
    logging.basicConfig(level=logging.NOTSET, format="[ %(asctime)s ]  %(levelname)s : %(message)s", datefmt="%d-%b-%y %H:%M:%S")
    logging.info("Loading data and modules ...")
    parser = argparse.ArgumentParser(description='Generating 4D maps for subjects')
    parser.add_argument('-d', '--domain', required=True, help='All brain networks in a specific domain')    
    parser.add_argument('-i', '--input', required=True, help='Directory of input Nifti image/images')    
    args = parser.parse_args()
    if torch.cuda.is_available():
        logging.debug("Available processing unit ({})".format(torch.cuda.get_device_name(0)))
    try:
        stored_files_list = []
        setting_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)) , "setting.json")
        input_data = os.path.abspath(args.input)
        if not (os.path.exists(setting_file_path)):
            raise FileNotFoundError("Setting file not found")
        if not (args.domain in ['ALL', 'SC', 'AU', 'SM', 'VI', 'CC', 'DM', 'CB']):
            raise ValueError("There is no such a brain domain!")
        if not (input_data.lower().endswith(('.nii', '.pkl'))):
            raise TypeError("Invalid input file type. Input should be a Nifti image or list of images in a pickle file!")
        if not (os.path.exists(input_data)):
            raise FileNotFoundError("Input image/table file not found, check your input arguments!")
        with open(setting_file_path) as file:
            conf = json.load(file)
        table_file_path = os.path.abspath(conf['models_table']['file_path'])
        saving_directory = os.path.abspath(conf['storage']['file_path'])
        mask_file_path = os.path.abspath(conf['mask']['file_path'])
        if not (os.path.exists(table_file_path)):
            raise FileNotFoundError("Table of pretrained models' information not found, check setting.json file!") 
        if not (os.path.exists(saving_directory)):
            raise FileNotFoundError("Map's saving directory not found, check setting.json file!")  
        if not (os.path.exists(mask_file_path)):
            raise FileNotFoundError("Mask image file not found, check setting.json file!")       
        network_list = pd.read_pickle(table_file_path)
        if args.domain != 'ALL':
            network_list = network_list.loc[network_list.domain == args.domain]
        if len(network_list) < 1:
            raise AssertionError("No records found for specific brain domain.")
        logging.info("Total number of {} brain networks are detected.".format(len(network_list)))
        for _,row in network_list.iterrows():
            m_time = time.time()
            base_model = load_model(row['modelPath'])
            logging.info("The {} from {} domain is loaded ...".format(row['networkId'], row['domain']))
            stored_files_list.append(predict_map(base_model, input_data, mask_file_path, row['networkId'], saving_directory))
            logging.info("{} runtime is {}".format(row['networkId'],(time.time() - m_time)))           
        res_file_dir = os.path.join(saving_directory, "result_{}_{}.txt".format(args.domain, time.strftime("%m%d%y_%H%M%S")))
        result_file = open(res_file_dir, "w")
        for element in stored_files_list:
            result_file.write(str(element) + "\n")
        result_file.close() 
    except Exception as e:
        logging.error("({}) - {}".format(e.__class__.__name__, e))
    logging.info("Map generation procedure is finished! You can find saved files' directories saved_directory/result_i.txt")


if __name__ == '__main__':
    main()

