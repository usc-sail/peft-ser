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

# from utils
from peft_ser.model.whisper import WhisperSER
from peft_ser.model.wavlm_plus import WavLMPlusSER
from peft_ser.model.wavlm_large import WavLMLargeSER
from peft_ser.model.mms import MMSSER
# from wavlm_plus import WavLMWrapper
# from wavlm_large import WavLMLargeWrapper
# from whisper import WhisperWrapper

# define logging console
import logging
logging.basicConfig(
    format='%(asctime)s %(levelname)-3s ==> %(message)s', 
    level=logging.INFO, 
    datefmt='%Y-%m-%d %H:%M:%S'
)

os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ["OMP_NUM_THREADS"] = "1" 

if __name__ == '__main__':

    # model_name = "wavlm_large"
    # model_name = "whisper_small"
    # model_name = "wavlm_plus"
    # model_name = "whisper_tiny"
    model_name = "mms-300m"
    lora_rank = 16
    
    if "whisper" in model_name:
        model = WhisperSER(
            pretrained_model=model_name,
            output_class_num=4,
            finetune_method="lora",
            lora_rank=lora_rank,
            cache_dir="/scratch1/tiantiaf/",
            use_conv_output=True,
            enable_peft_training=True
        )
    elif model_name == "wavlm_plus":
        model = WavLMPlusSER(
            pretrained_model=model_name,
            output_class_num=4,
            finetune_method="lora",
            lora_rank=lora_rank,
            cache_dir="/scratch1/tiantiaf/",
            use_conv_output=True,
            enable_peft_training=True
        )
    elif model_name == "wavlm_large":
        model = WavLMLargeSER(
            pretrained_model=model_name,
            output_class_num=4,
            finetune_method="lora",
            lora_rank=lora_rank,
            cache_dir="/scratch1/tiantiaf/",
            use_conv_output=True,
            enable_peft_training=True
        )
    elif model_name == "mms-300m":
        model = MMSSER(
            pretrained_model=model_name,
            output_class_num=4,
            finetune_method="lora",
            lora_rank=lora_rank,
            cache_dir="/scratch1/tiantiaf/",
            use_conv_output=True,
            enable_peft_training=True
        )
    # f"/project/shrikann_35/tiantiaf/effcient-ser/finetune/{model_name}/lr000025_ep30_{model.model_setting}"
    # /project/shrikann_35/tiantiaf/effcient-ser/finetune/
    model_path = f"/project/shrikann_35/tiantiaf/effcient-ser/finetune/emo4/{model_name}/lr000025_ep30_{model.model_setting}/fold_1.pt"
    lora_model_path = f"/project/shrikann_35/tiantiaf/effcient-ser/finetune/emo4/{model_name}/lr000025_ep30_{model.model_setting}/lora_fold_1.pt"
    
    save_path = f"/project/shrikann_35/tiantiaf/effcient-ser/finetune/release/{model_name}_{model.model_setting}.pt"
    lora_save_path = f"/project/shrikann_35/tiantiaf/effcient-ser/finetune/release/{model_name}_{model.model_setting}_lora.pt"
    weights = torch.load(model_path)
    weights["layer_weights"] = weights["weights"]
    del weights["weights"]
    
    filter_weights = {k: v for k, v in weights.items() if 'backbone_model' not in k and "embed_positions" not in k}
    # model.load_state_dict(torch.load(model_path))
    lora_weights = torch.load(lora_model_path)
    pdb.set_trace()
    
    torch.save(filter_weights, save_path)
    torch.save(lora_weights, lora_save_path)
    # filter_weights.keys()
    # pdb.set_trace()