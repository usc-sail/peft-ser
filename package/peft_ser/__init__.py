import os
import pdb
import torch

import urllib
from tqdm import tqdm
from pathlib import Path

from .model.whisper import WhisperSER
from .model.wavlm_plus import WavLMPlusSER
from .model.wavlm_large import WavLMLargeSER
from .model.mms import MMSSER


model_config = {
    "whisper-base-lora-16-conv": {
        "pretrained_model": "whisper_base",
        "finetune_method": "lora",
        "lora_rank": 16,
        "use_conv_output": True,
        "classifier_link": "https://www.dropbox.com/scl/fi/sq6w9vfrlyd0t3y5xwnnm/whisper_base_lora_16_conv_output.pt?rlkey=xtot2gi9au83xi4f3kxzwjac9&dl=1",
        "lora_link": "https://www.dropbox.com/scl/fi/46bb08ek2zbt32fmqrg21/whisper_base_lora_16_conv_output_lora.pt?rlkey=452zdg99to7ntol3rjrtgccog&dl=1"
    },
    "whisper-small-lora-16-conv": {
        "pretrained_model": "whisper_small",
        "finetune_method": "lora",
        "lora_rank": 16,
        "use_conv_output": True,
        "classifier_link": "https://www.dropbox.com/scl/fi/q99rdffwnun4f62ksr6wl/whisper_small_lora_16_conv_output.pt?rlkey=1jb66t26cgh18wi4zd8yhi27s&dl=1",
        "lora_link": "https://www.dropbox.com/scl/fi/deq7cyylglk9ztf5gqr7i/whisper_small_lora_16_conv_output_lora.pt?rlkey=98ub4alelsyw5gddw6k1nwqn7&dl=1"
    },
    "whisper-tiny-lora-16-conv": {
        "pretrained_model": "whisper_tiny",
        "finetune_method": "lora",
        "lora_rank": 16,
        "use_conv_output": True,
        "classifier_link": "https://www.dropbox.com/scl/fi/ld0pymoj8iiiaen9qgpom/whisper_tiny_lora_16_conv_output.pt?rlkey=7sum57se4fmdqwhvte7eiexkc&dl=1",
        "lora_link": "https://www.dropbox.com/scl/fi/v9t8eyvsq882a0ulht80h/whisper_tiny_lora_16_conv_output_lora.pt?rlkey=1c3uzxifkm7788q61lnmkrevt&dl=1"
    },
    "wavlm-large-lora-16-conv": {
        "pretrained_model": "wavlm_large",
        "finetune_method": "lora",
        "lora_rank": 16,
        "use_conv_output": True,
        "classifier_link": "https://www.dropbox.com/scl/fi/1br3nrrnfmn0pd3kpmp5s/wavlm_large_lora_16_conv_output.pt?rlkey=jnghxe6mcz9cw78kai09k4yex&dl=1",
        "lora_link": "https://www.dropbox.com/scl/fi/4ybo1iu62t178albsbpcv/wavlm_large_lora_16_conv_output_lora.pt?rlkey=bjpjv0s66hl8z3bngzgtfzqds&dl=1"
    },
    "wavlm-plus-lora-16-conv": {
        "pretrained_model": "wavlm_plus",
        "finetune_method": "lora",
        "lora_rank": 16,
        "use_conv_output": True,
        "classifier_link": "https://www.dropbox.com/scl/fi/zlhdeoxzweksxt59d7jss/wavlm_plus_lora_16_conv_output.pt?rlkey=5trov7d6ki48bwcl25e63ffko&dl=1",
        "lora_link": "https://www.dropbox.com/scl/fi/19sb7huz9cknlybumjnfv/wavlm_plus_lora_16_conv_output_lora.pt?rlkey=8dlc3vei5cdir24wv1h1av187&dl=1"
    },
    "mms-lora-16-conv": {
        "pretrained_model": "mms-300m",
        "finetune_method": "lora",
        "lora_rank": 16,
        "use_conv_output": True,
        "classifier_link": "https://www.dropbox.com/scl/fi/mptq3vpu5b71o9azqvw8d/mms-300m_lora_16_conv_output.pt?rlkey=sj09sgy8sri0faz2eb194orwf&dl=1",
        "lora_link": "https://www.dropbox.com/scl/fi/z6uib8ivqlfty7wla8wjq/mms-300m_lora_16_conv_output_lora.pt?rlkey=nc46hvby59o1on8hiomitc916&dl=1"
    }
}

def _download_from_url(url: str, download_target: str):
    # os.makedirs(root, exist_ok=True)
    # download_target = os.path.join(root, os.path.basename(url))
    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(f"{download_target} exists and is not a regular file")
        
    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
        with tqdm(
            total=int(source.info().get("Content-Length")),
            ncols=80,
            unit="iB",
            unit_scale=True,
            unit_divisor=1024,
        ) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break
                output.write(buffer)
                loop.update(len(buffer))

def load_model(pretrained_ser_model, cache_folder: str=str(Path.cwd())):
    # /scratch1/tiantiaf/test
    # whisper_base_lora_16_conv_output
    # ser_model.load_model("whisper-base-lora-16-conv", cache_folder="/scratch1/tiantiaf/test")
    assert pretrained_ser_model in model_config.keys(), "Invalid model name"
    
    model_name = model_config[pretrained_ser_model]["pretrained_model"]
    classifier_url = model_config[pretrained_ser_model]["classifier_link"]
    lora_url = model_config[pretrained_ser_model]["lora_link"]
    finetune_method = model_config[pretrained_ser_model]["finetune_method"]
    lora_rank = model_config[pretrained_ser_model]["lora_rank"]
    use_conv_output = model_config[pretrained_ser_model]["use_conv_output"]
    
    if "whisper" in model_name:
        model = WhisperSER(
            pretrained_model=model_name,
            output_class_num=4,
            finetune_method=finetune_method,
            lora_rank=lora_rank,
            cache_dir=cache_folder,
            use_conv_output=use_conv_output,
            enable_peft_training=False
        )
    elif model_name == "wavlm_plus":
        model = WavLMPlusSER(
            pretrained_model=model_name,
            output_class_num=4,
            finetune_method=finetune_method,
            lora_rank=lora_rank,
            cache_dir=cache_folder,
            use_conv_output=use_conv_output,
            enable_peft_training=True
        )
    elif model_name == "wavlm_large":
        model = WavLMLargeSER(
            pretrained_model=model_name,
            output_class_num=4,
            finetune_method=finetune_method,
            lora_rank=lora_rank,
            cache_dir=cache_folder,
            use_conv_output=use_conv_output,
            enable_peft_training=True
        )
    elif model_name == "mms-300m":
        model = MMSSER(
            pretrained_model=model_name,
            output_class_num=4,
            finetune_method="lora",
            lora_rank=lora_rank,
            cache_dir=cache_folder,
            use_conv_output=True,
            enable_peft_training=True
        )
    # Load classifier
    save_path = str(Path.joinpath(Path(cache_folder), f"{pretrained_ser_model}.pt"))
    if not Path.exists(Path(save_path)):
        _download_from_url(url=classifier_url, download_target=save_path)
    with (open(save_path, "rb")) as fp: 
        checkpoint = torch.load(fp)
    del save_path
    model.load_state_dict(checkpoint, strict=False)

    # Load LoRa
    if finetune_method == "lora":
        lora_path = str(Path.joinpath(Path(cache_folder), f"{pretrained_ser_model}_lora.pt"))
        if not Path.exists(Path(lora_path)):
            _download_from_url(url=lora_url, download_target=lora_path)
        with (open(lora_path, "rb")) as fp: 
            checkpoint = torch.load(fp)
        model.load_state_dict(checkpoint, strict=False)
    return model