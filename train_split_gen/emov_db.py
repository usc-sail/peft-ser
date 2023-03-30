import json
import numpy as np
import pandas as pd
import pickle, pdb, re, os

from tqdm import tqdm
from pathlib import Path


# define logging console
import logging
logging.basicConfig(
    format='%(asctime)s %(levelname)-3s ==> %(message)s', 
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)


if __name__ == '__main__':

    # data path
    data_path = Path('/media/data/public-data/SER/EmoV-db/emov_db')
    output_path = Path('/media/data/projects/speech-privacy/trust-ser')

    Path.mkdir(output_path.joinpath('train_split'), parents=True, exist_ok=True)
    train_list, dev_list, test_list = list(), list(), list()
    
    # vox-movie
    for speaker_id_emo in os.listdir(data_path):
        if '.' in speaker_id_emo: continue
        for file_name in os.listdir(data_path.joinpath(speaker_id_emo)):
            # [key, speaker id, gender, path, label]
            file_path = data_path.joinpath(speaker_id_emo, file_name)
            file_data = [file_name, f'emov_db_{speaker_id_emo.split("_")[0]}', '', str(file_path), speaker_id_emo.split('_')[1]]
            train_list.append(file_data)
    
    return_dict = dict()
    return_dict['train'], return_dict['dev'], return_dict['test'] = train_list, dev_list, test_list
    logging.info(f'-------------------------------------------------------')
    logging.info(f'Split distribution for EmoV-db dataset')
    for split in ['train', 'dev', 'test']:
        logging.info(f'Split {split}: Number of files {len(return_dict[split])}')
    logging.info(f'-------------------------------------------------------')
    
    # dump the dictionary
    jsonString = json.dumps(return_dict, indent=4)
    jsonFile = open(str(output_path.joinpath('train_split', f'emov_db.json')), "w")
    jsonFile.write(jsonString)
    jsonFile.close()

    