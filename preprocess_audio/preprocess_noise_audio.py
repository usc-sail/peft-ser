import json
import glob
import torch
import torchaudio
import numpy as np
import pandas as pd
import pickle, pdb, re

from tqdm import tqdm
from pathlib import Path
from moviepy.editor import *

# define logging console
import logging
logging.basicConfig(
    format='%(asctime)s %(levelname)-3s ==> %(message)s', 
    level=logging.INFO, 
    datefmt='%Y-%m-%d %H:%M:%S'
)


if __name__ == '__main__':

    # data path
    audio_path = Path('/media/data/projects/speech-privacy/emo2vec/noise_audio')

    # noise files
    musan_wav_files = glob.glob("/media/data/public-data/AudioNoise/musan/musan/noise/*/*.wav")
    esc_wav_files = glob.glob("/media/data/public-data/AudioNoise/esc-50/ESC-50-master/audio/*.wav")

    noise_wav_files = list()
    for file_path in musan_wav_files: noise_wav_files.append(file_path)
    for file_path in esc_wav_files: noise_wav_files.append(file_path)

    Path.mkdir(audio_path, parents=True, exist_ok=True)
    for noise_wav_file in tqdm(noise_wav_files, ncols=100, miniters=100):
        waveform, sample_rate = torchaudio.load(str(noise_wav_file))
        if "musan" in str(noise_wav_file): dataset = "musan"
        else: dataset = "esc-50"
        
        if waveform.shape[0] != 1:
            waveform = torch.mean(waveform, dim=0).unsqueeze(0)
        if sample_rate != 16000:
            transform_model = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = transform_model(waveform)

        output_path = audio_path.joinpath(f'{dataset}_{str(noise_wav_file).split("/")[-1]}')
        torchaudio.save(str(output_path), waveform, 16000)
           