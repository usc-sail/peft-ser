import json
import yaml
import numpy as np
import pandas as pd
import pickle, pdb, re, os

from tqdm import tqdm
from pathlib import Path
from sklearn.model_selection import KFold

# define logging console
import logging
logging.basicConfig(
    format='%(asctime)s %(levelname)-3s ==> %(message)s', 
    level=logging.INFO, 
    datefmt='%Y-%m-%d %H:%M:%S'
)


if __name__ == '__main__':

    # Read data path
    with open("../config/config.yml", "r") as stream:
        config = yaml.safe_load(stream)
    data_path   = Path(config["data_dir"]["ravdess"])
    output_path = Path(config["project_dir"])

    # ravdess speakers
    speaker_ids = [speaker_id for speaker_id in os.listdir(data_path) if '.zip' not in speaker_id]
    kf = KFold(n_splits=5, random_state=None, shuffle=False)
    for fold_idx, (train_index, test_index) in enumerate(kf.split(speaker_ids)):
        Path.mkdir(output_path.joinpath('train_split'), parents=True, exist_ok=True)
        train_list, dev_list, test_list = list(), list(), list()

        train_index, dev_index = train_index[:-len(train_index)//5], train_index[-len(train_index)//5:]
        # read speakers
        train_speakers = [np.arange(1, len(speaker_ids)+1, 1)[idx] for idx in train_index]
        dev_speakers = [np.arange(1, len(speaker_ids)+1, 1)[idx] for idx in dev_index]
        test_speakers = [np.arange(1, len(speaker_ids)+1, 1)[idx] for idx in test_index]

        for speaker_id in speaker_ids:
            for file_name in os.listdir(data_path.joinpath(speaker_id)):
                actor_id = int(speaker_id.split('_')[1])
                label = int(file_name.split('-')[2])
                gender = 'female' if int(file_name.split('-')[-1].split('.')[0]) % 2 else 'male'
                file_path = data_path.joinpath(speaker_id, file_name)
                # [key, speaker id, gender, path, label]
                file_data = [file_name, f'ravdess_{speaker_id}', gender, str(file_path), label]
                
                # append data
                if actor_id in test_speakers: test_list.append(file_data)
                elif actor_id in dev_speakers: dev_list.append(file_data)
                else: train_list.append(file_data)
        
        return_dict = dict()
        return_dict['train'], return_dict['dev'], return_dict['test'] = train_list, dev_list, test_list
        logging.info(f'-------------------------------------------------------')
        logging.info(f'Split distribution for RAVDESS dataset')
        for split in ['train', 'dev', 'test']: logging.info(f'Split {split}: Number of files {len(return_dict[split])}')
        logging.info(f'-------------------------------------------------------')
        
        # dump the dictionary
        jsonString = json.dumps(return_dict, indent=4)
        jsonFile = open(str(output_path.joinpath('train_split', f'ravdess_fold{fold_idx+1}.json')), "w")
        jsonFile.write(jsonString)
        jsonFile.close()

    