import json
import yaml
import torch
import random
import numpy as np
import pandas as pd
import torch.nn as nn
import loralib as lora
import argparse, logging
import torch.multiprocessing
import copy, time, pickle, shutil, sys, os, pdb

from tqdm import tqdm
from pathlib import Path
from copy import deepcopy
from collections import defaultdict, deque
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau


sys.path.append(os.path.join(str(Path(os.path.realpath(__file__)).parents[1])))
sys.path.append(os.path.join(str(Path(os.path.realpath(__file__)).parents[1]), 'model'))
sys.path.append(os.path.join(str(Path(os.path.realpath(__file__)).parents[1]), 'dataloader'))

from utils import parse_finetune_args, set_seed, log_epoch_result, log_best_result

# from utils
from wav2vec import Wav2VecWrapper
from wavlm_plus import WavLMWrapper
from whisper import WhisperWrapper
from evaluation import EvalMetric
from dataloader import load_finetune_audios, set_finetune_dataloader, return_weights


# define logging console
import logging
logging.basicConfig(
    format='%(asctime)s %(levelname)-3s ==> %(message)s', 
    level=logging.INFO, 
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Model hidden states information
hid_dim_dict = {
    "wav2vec2_0":       768,
    "tera":             768,
    "wavlm":            768,
    "whisper_small":    768,
    "whisper_base":     512,
    "whisper_tiny":     384,
    "apc":              512,
}

# Model number of encoding layers
num_enc_layers_dict = {
    "wav2vec2_0":       12,
    "wavlm":            12,
    "whisper_small":    12,
    "whisper_base":     6,
    "tera":             4,
    "whisper_tiny":     4,
    "apc":              3,
}

os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ["OMP_NUM_THREADS"] = "1" 

if __name__ == '__main__':

    # Argument parser
    args = parse_finetune_args()
    with open("../config/config.yml", "r") as stream: config = yaml.safe_load(stream)
    args.split_dir  = str(Path(config["project_dir"]).joinpath("train_split"))
    args.data_dir   = str(Path(config["project_dir"]).joinpath("audio"))
    args.log_dir    = str(Path(config["project_dir"]).joinpath("finetune"))

    # Find device
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available(): print('GPU available, use GPU')
    
    for pretrained_model in ["wav2vec2_0", "wavlm_plus", "whisper_tiny", "whisper_base", "whisper_small"]:
        args.pretrain_model = pretrained_model
        for finetune_method in ["finetune", "adapter", "embedding_prompt", "lora"]:
            if finetune_method == "adapter":
                args.finetune_method        = "adapter"
                args.adapter_hidden_dim     = 128
            elif finetune_method == "embedding_prompt":
                args.finetune_method        = "embedding_prompt"
                args.embedding_prompt_dim   = 5
            elif finetune_method == "lora":
                args.finetune_method        = "lora"
                args.lora_rank              = 32
            else:
                args.finetune_method        = "finetune"
            
            # Define the model wrapper
            if args.pretrain_model == "wav2vec2_0":
                # Wav2vec2_0 Wrapper
                model = Wav2VecWrapper(args).to(device)
            elif args.pretrain_model == "wavlm_plus":
                # WavLM Plus Wrapper
                model = WavLMWrapper(args).to(device)
            elif args.pretrain_model in ["whisper_tiny", "whisper_base", "whisper_small", "whisper_medium", "whisper_large"]:
                # Whisper Plus Wrapper
                model = WhisperWrapper(args).to(device)
            
            # Read trainable params
            if args.finetune_method == "finetune":
                model_parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
            else:
                model_parameters = list(filter(lambda p: p.requires_grad, model.backbone_model.parameters()))
            params = sum([np.prod(p.size()) for p in model_parameters])
            logging.info(f'{args.pretrain_model}, {args.finetune_method}, Trainable params size: {params/(1e6):.2f} M')
        