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

def read_speaker(folder_path: str, split: str = 'train') -> (dict, int):
    """
    Gets wav data dict and the number of unique classes for task.
    :param folder_path: Folder path in which corresponding download script for meld dataset download_audio.sh was executed
    :param split: split to load. Either 'train','test' or 'dev'
    :return: data_dict with structure {speaker_index:[[key, file_path, category] ... ] and number of classes
    """
    if split not in ['train', 'test', 'dev']:
        raise ValueError("split must be either 'train','test' or 'dev' for MELD dataset")
        
    if split == 'train':
        label_path = f'{folder_path}/train_sent_emo.csv'
        data_path = f'{folder_path}/train_splits'
    elif split == 'test':
        label_path = f'{folder_path}/test_sent_emo.csv'
        data_path = f'{folder_path}/output_repeated_splits_test'
    elif split == 'dev':
        label_path = f'{folder_path}/dev_sent_emo.csv'
        data_path = f'{folder_path}/dev_splits_complete'
        
    df_label = pd.read_csv(label_path)
    err = []
    for i, df_row in tqdm(df_label.iterrows()):
        if not Path(f"{data_path}/waves/dia{df_row.Dialogue_ID}_utt{df_row.Utterance_ID}.wav").is_file():
            err.append(i)
    print(f'Missing/Corrupt files for indices: {err}')
    df_label_cleaned = df_label.drop(err)
        
    df_label_cleaned['Path'] = df_label_cleaned.apply(lambda row: f"{data_path}/waves/dia{row.Dialogue_ID}_utt{row.Utterance_ID}.wav", axis=1)
    df_label_cleaned['Filename'] = df_label_cleaned.apply(lambda row: f"dia{row.Dialogue_ID}_utt{row.Utterance_ID}", axis=1)
    
    df_label_reduced = df_label_cleaned[['Speaker', 'Filename', 'Path', 'Emotion']]
    groups = df_label_reduced.groupby('Speaker')
    data_dict = {speaker: group[['Filename', 'Path', 'Emotion']].values.tolist() for _, (speaker, group) in enumerate(groups)}
    
    return_list = list()
    for speaker_id in data_dict:
        for idx in range(len(data_dict[speaker_id])):
            file_name = data_dict[speaker_id][idx][0]
            file_path = data_dict[speaker_id][idx][1]
            label = data_dict[speaker_id][idx][2]
            return_list.append([file_name, f'meld_{speaker_id}', '', str(file_path), label])
    return return_list


if __name__ == '__main__':

    # Read data path
    with open("../config/config.yml", "r") as stream:
        config = yaml.safe_load(stream)
    data_path   = Path(config["data_dir"]["meld"])
    output_path = Path(config["project_dir"])

    Path.mkdir(output_path.joinpath('train_split'), parents=True, exist_ok=True)
    train_list, dev_list, test_list = list(), list(), list()
    
    # data root folder
    train_list = read_speaker(data_path, 'train')
    dev_list = read_speaker(data_path, 'dev')
    test_list = read_speaker(data_path, 'test')
        
    return_dict = dict()
    return_dict['train'], return_dict['dev'], return_dict['test'] = train_list, dev_list, test_list
    logging.info(f'-------------------------------------------------------')
    logging.info(f'Split distribution for MELD dataset')
    for split in ['train', 'dev', 'test']:
        logging.info(f'Split {split}: Number of files {len(return_dict[split])}')
    logging.info(f'-------------------------------------------------------')
        
    # dump the dictionary
    jsonString = json.dumps(return_dict, indent=4)
    jsonFile = open(str(output_path.joinpath('train_split', f'meld.json')), "w")
    jsonFile.write(jsonString)
    jsonFile.close()

    