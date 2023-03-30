import json
import torch
import torchaudio
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

    # Read configs
    with open("../config/config.yml", "r") as stream:
         config = yaml.safe_load(stream)
    data_path   = Path(config["data_dir"]["crema_d"])
    split_path = Path(config["project_dir"]).joinpath("train_split")
    audio_path = Path(config["project_dir"]).joinpath("audio")

    # Parser the dataset
    parser = argparse.ArgumentParser(description='input dataset')
    parser.add_argument(
        '--dataset', 
        default='iemocap',
        type=str, 
        help='dataset name'
    )
    args = parser.parse_args()

    # Read split file
    if args.dataset in ["iemocap", "crema_d", "ravdess", "msp-improv"]:
        with open(str(split_path.joinpath(f'{args.dataset}_fold1.json')), "r") as f: split_dict = json.load(f)
    else:
        with open(str(split_path.joinpath(f'{args.dataset}.json')), "r") as f: split_dict = json.load(f)

    # Iterate over different splits
    for split in ['train', 'dev', 'test']:
        Path.mkdir(audio_path.joinpath(args.dataset), parents=True, exist_ok=True)
        for idx in tqdm(range(len(split_dict[split]))):
            # Read data: speaker_id, path
            data = split_dict[split][idx]
            speaker_id, file_path = data[1], data[3]

            # Read wavforms
            waveform, sample_rate = torchaudio.load(str(file_path))

            # If the waveform has multiple channels, compute the mean across channels to create a single-channel waveform.
            if waveform.shape[0] != 1:
                waveform = torch.mean(waveform, dim=0).unsqueeze(0)

            # If the sample rate is not 16000 Hz, resample the waveform to 16000 Hz.
            if sample_rate != 16000:
                transform_model = torchaudio.transforms.Resample(sample_rate, 16000)
                waveform = transform_model(waveform)

            # If the dataset is "cmu-mosei", extract a specific segment of the waveform based on time information.
            if args.dataset == 'cmu-mosei':
                start, end = int(data[4][0] * 16000), int(data[4][1] * 16000)
                waveform = waveform[:, start:end]

            # Set the output path for the processed audio file based on the dataset and other information.
            if args.dataset in ['iemocap', 'msp-improv', 'meld', 'crema_d', 'msp-podcast']:
                output_path = audio_path.joinpath(args.dataset, file_path.split('/')[-1])
            elif args.dataset in ['ravdess', 'emov_db', 'vox-movie']:
                output_path = audio_path.joinpath(args.dataset, f'{speaker_id}_{file_path.split("/")[-1]}')
            elif args.dataset in ['cmu-mosei']:
                output_path = audio_path.joinpath(args.dataset, f'{str(start).replace(".", "_")}_{file_path.split("/")[-1]}')
            
            # Save the audio file with desired sampling frequency
            torchaudio.save(str(output_path), waveform, 16000)
            split_dict[split][idx][3] = str(output_path)

        # Logging the stats for train/dev/test
        logging.info(f'-------------------------------------------------------')
        logging.info(f'Preprocess audio for {dataset} dataset')
        for split in ['train', 'dev', 'test']: logging.info(f'Split {split}: Number of files {len(split_dict[split])}')
        logging.info(f'-------------------------------------------------------')
