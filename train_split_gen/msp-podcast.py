import json
import yaml
import numpy as np
import pandas as pd
import pickle, pdb, re

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

    # Read data path
    with open("../config/config.yml", "r") as stream:
        config = yaml.safe_load(stream)
    data_path   = Path(config["data_dir"]["msp-podcast"])
    output_path = Path(config["project_dir"])

    Path.mkdir(output_path.joinpath('train_split'), parents=True, exist_ok=True)
    train_list, dev_list, test_list = list(), list(), list()
    
    # msp-podcast
    label_dict = dict()
    
    # iterate over the labels
    label_df = pd.read_csv(Path(data_path).joinpath("Labels", "labels_concensus.csv"), index_col=0)
    for idx, file_name in tqdm(enumerate(list(label_df.index)[:]), ncols=100, miniters=100):
        file_path = Path(data_path).joinpath("Audios", file_name)
        if Path.exists(file_path) is False: continue
        
        speaker_id = label_df.loc[file_name, "SpkrID"]
        gender = label_df.loc[file_name, "Gender"]
        label = label_df.loc[file_name, "EmoClass"]
        arousal = label_df.loc[file_name, "EmoAct"]
        valence = label_df.loc[file_name, "EmoVal"]

        # skip condictions, unknown speakers, other emotions, no agreement emotions
        if "known" in speaker_id or "known" in gender: continue
        session_id = label_df.loc[file_name, "Split_Set"]
        
        # [key, speaker id, gender, path, label]
        gender_label = "female" if gender == "Female" else "male"
        file_data = [file_name, f'msp_podcast_{speaker_id}', gender_label, str(file_path), label]

        # append data
        if session_id == 'Test1': test_list.append(file_data)
        # elif session_id == 'Test2': test2_list.append(file_data)
        elif session_id == 'Validation': dev_list.append(file_data)
        elif session_id == 'Train': train_list.append(file_data)
        
    return_dict = dict()
    # return_dict['train'], return_dict['dev'], return_dict['test1'], return_dict['test2'] = train_list, dev_list, test1_list, test2_list
    return_dict['train'], return_dict['dev'], return_dict['test'] = train_list, dev_list, test_list
    logging.info(f'-------------------------------------------------------')
    logging.info(f'Split distribution for MSP-podcast dataset')
    # for split in ['train', 'dev', 'test1', 'test2']:
    for split in ['train', 'dev', 'test']: logging.info(f'Split {split}: Number of files {len(return_dict[split])}')
    logging.info(f'-------------------------------------------------------')
    
    # dump the dictionary
    jsonString = json.dumps(return_dict, indent=4)
    jsonFile = open(str(output_path.joinpath('train_split', f'msp-podcast.json')), "w")
    jsonFile.write(jsonString)
    jsonFile.close()

    