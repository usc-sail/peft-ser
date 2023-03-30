import json
import yaml
import numpy as np
import pandas as pd
import pickle, pdb, re

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
    data_path   = Path(config["data_dir"]["crema_d"])
    output_path = Path(config["project_dir"])
    
    kf = KFold(n_splits=5, random_state=None, shuffle=False)
    for fold_idx, (train_index, test_index) in enumerate(kf.split(np.arange(1001, 1092, 1))):
        Path.mkdir(output_path.joinpath('train_split'), parents=True, exist_ok=True)
        train_list, dev_list, test_list = list(), list(), list()
        
        # crema-d
        file_list = [x for x in Path(data_path).joinpath("AudioWAV").iterdir() if '.wav' in x.parts[-1]]
        file_list.sort()
        # read demographics and ratings
        demo_df = pd.read_csv(str(Path(data_path).joinpath('VideoDemographics.csv')), index_col=0)
        rating_df = pd.read_csv(str(Path(data_path).joinpath('processedResults', 'summaryTable.csv')), index_col=1)
        train_index, dev_index = train_index[:-len(train_index)//5], train_index[-len(train_index)//5:]
        # read speakers
        train_speakers = [np.arange(1001, 1092, 1)[idx] for idx in train_index]
        dev_speakers = [np.arange(1001, 1092, 1)[idx] for idx in dev_index]
        test_speakers = [np.arange(1001, 1092, 1)[idx] for idx in test_index]
        
        for idx, file_path in enumerate(file_list):
            # read basic information
            if '1076_MTI_SAD_XX.wav' in str(file_path): continue
            sentence_file = file_path.parts[-1].split('.wav')[0]
            sentence_part = sentence_file.split('_')
            speaker_id = int(sentence_part[0])
            gender = 'male' if demo_df.loc[int(speaker_id), 'Sex'] == 'Male' else 'female'
            label = rating_df.loc[sentence_file, 'MultiModalVote']
            
            # [key, speaker id, gender, path, label]
            file_data = [sentence_file, speaker_id, gender, str(file_path), label]

            # append data
            if speaker_id in test_speakers: test_list.append(file_data)
            elif speaker_id in dev_speakers: dev_list.append(file_data)
            else: train_list.append(file_data)
        
        return_dict = dict()
        return_dict['train'], return_dict['dev'], return_dict['test'] = train_list, dev_list, test_list
        logging.info(f'-------------------------------------------------------')
        logging.info(f'Split distribution for CREMA-D dataset')
        for split in ['train', 'dev', 'test']:
            logging.info(f'Split {split}: Number of files {len(return_dict[split])}')
        logging.info(f'-------------------------------------------------------')
        # dump the dictionary
        jsonString = json.dumps(return_dict, indent=4)
        jsonFile = open(str(output_path.joinpath('train_split', f'crema_d_fold{fold_idx+1}.json')), "w")
        jsonFile.write(jsonString)
        jsonFile.close()

    