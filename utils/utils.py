import json
import torch
import random
import numpy as np
import argparse, logging

logging.basicConfig(
    format='%(asctime)s %(levelname)-3s ==> %(message)s', 
    level=logging.INFO, 
    datefmt='%Y-%m-%d %H:%M:%S'
)

def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    
def get_results(input_dict):
    return_dict = dict()
    return_dict["uar"] = input_dict["uar"]
    return_dict["acc"] = input_dict["acc"]
    return_dict["loss"] = input_dict["loss"]
    return return_dict

def log_epoch_result(
    result_hist_dict:       dict, 
    epoch:                  int,
    train_result:           dict,
    dev_result:             dict,
    test_result:            dict,
    log_dir:                str,
    fold_idx:               int
):
    # read result
    result_hist_dict[epoch] = dict()
    result_hist_dict[epoch]["train"] = get_results(train_result)
    result_hist_dict[epoch]["dev"] = get_results(dev_result)
    result_hist_dict[epoch]["test"] = get_results(test_result)
    
    # dump the dictionary
    jsonString = json.dumps(result_hist_dict, indent=4)
    jsonFile = open(str(log_dir.joinpath(f'fold_{fold_idx}.json')), "w")
    jsonFile.write(jsonString)
    jsonFile.close()
    
    
def log_best_result(
    result_hist_dict:       dict, 
    epoch:                  int,
    best_dev_uar:           float,
    best_dev_acc:           float,
    best_test_uar:          float,
    best_test_acc:          float,
    log_dir:                str,
    fold_idx:               int
):
    # log best result
    result_hist_dict["best"] = dict()
    result_hist_dict["best"]["dev"], result_hist_dict["best"]["test"] = dict(), dict()
    result_hist_dict["best"]["dev"]["uar"] = best_dev_uar
    result_hist_dict["best"]["dev"]["acc"] = best_dev_acc
    result_hist_dict["best"]["test"]["uar"] = best_test_uar
    result_hist_dict["best"]["test"]["acc"] = best_test_acc

    # save results for this fold
    jsonString = json.dumps(result_hist_dict, indent=4)
    jsonFile = open(str(log_dir.joinpath(f'fold_{fold_idx}.json')), "w")
    jsonFile.write(jsonString)
    jsonFile.close()

def parse_finetune_args():
    # parser
    parser = argparse.ArgumentParser(description='emo2vec finetune experiments')
    parser.add_argument(
        '--data_dir', 
        default='/media/data/projects/speech-privacy/trust-ser/audio',
        type=str, 
        help='raw audio path'
    )

    parser.add_argument(
        '--model_dir', 
        default='/media/data/projects/speech-privacy/trust-ser/model',
        type=str, 
        help='model save path'
    )

    parser.add_argument(
        '--split_dir', 
        default='/media/data/projects/speech-privacy/trust-ser/train_split',
        type=str, 
        help='train split path'
    )

    parser.add_argument(
        '--log_dir', 
        default='/media/data/projects/speech-privacy/trust-ser/finetune',
        type=str, 
        help='model save path'
    )

    parser.add_argument(
        '--uar_dir', 
        default='/media/data/projects/speech-privacy/trust-ser/uar',
        type=str, 
        help='model uar history'
    )

    parser.add_argument(
        '--attack_dir', 
        default='/media/data/projects/speech-privacy/trust-ser/attack',
        type=str, 
        help='attack data'
    )
    
    parser.add_argument(
        '--privacy_attack_dir', 
        default='/media/data/projects/speech-privacy/trust-ser/privacy',
        type=str, 
        help='privacy attack method data'
    )
    
    parser.add_argument(
        '--privacy_attack', 
        default='gender',
        type=str, 
        help='Privacy attack method'
    )

    parser.add_argument(
        '--fairness_dir', 
        default='/media/data/projects/speech-privacy/trust-ser/fairness',
        type=str, 
        help='model save path'
    )

    parser.add_argument(
        '--sustainability_dir', 
        default='/media/data/projects/speech-privacy/trust-ser/sustainability',
        type=str, 
        help='model save path'
    )
    
    parser.add_argument(
        '--attack_method', 
        default='pgd',
        type=str, 
        help='attack method'
    )

    parser.add_argument(
        '--pretrain_model', 
        default='wav2vec2_0',
        type=str,
        help="pretrained model type"
    )

    parser.add_argument(
        '--finetune', 
        default='frozen',
        type=str,
        help="partial finetune or not"
    )
    
    parser.add_argument(
        '--learning_rate', 
        default=0.0002,
        type=float,
        help="learning rate",
    )

    parser.add_argument(
        '--num_epochs', 
        default=50,
        type=int,
        help="total training rounds",
    )
    
    parser.add_argument(
        '--optimizer', 
        default='adam',
        type=str,
        help="optimizer",
    )
    
    parser.add_argument(
        '--dataset',
        default="iemocap",
        type=str,
        help="Dataset name",
    )
    
    parser.add_argument(
        '--audio_duration', 
        default=6,
        type=int,
        help="audio length for training"
    )

    parser.add_argument(
        '--downstream_model', 
        default='rnn',
        type=str,
        help="model type"
    )

    parser.add_argument(
        '--num_layers',
        default=1,
        type=int,
        help="num of layers",
    )

    parser.add_argument(
        '--snr',
        default=45,
        type=int,
        help="SNR of the audio",
    )

    parser.add_argument(
        '--conv_layers',
        default=3,
        type=int,
        help="num of conv layers",
    )

    parser.add_argument(
        '--hidden_size',
        default=256,
        type=int,
        help="hidden size",
    )

    parser.add_argument(
        '--pooling',
        default='att',
        type=str,
        help="pooling method: att, average",
    )

    parser.add_argument(
        '--norm',
        default='nonorm',
        type=str,
        help="normalization or not",
    )
    
    args = parser.parse_args()
    return args

