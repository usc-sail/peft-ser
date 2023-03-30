# part of the code was referenced from SUPERB: https://github.com/s3prl/s3prl
import os
import pdb
import copy
import torch
import librosa
import itertools
import numpy as np
import s3prl.hub as hub
from functools import lru_cache
from torchaudio.compliance import kaldi

from torch import nn
from collections import OrderedDict
from typing import Optional, Callable
from torch.nn import functional as F
from torch.nn.functional import normalize
from transformers import Wav2Vec2Model, Wav2Vec2Config, Wav2Vec2Processor, AutoProcessor, WavLMModel, WhisperModel, AutoFeatureExtractor


@lru_cache(maxsize=None)
def mel_filters(device, n_mels: int = 80) -> torch.Tensor:
    """
    load the mel filterbank matrix for projecting STFT into a Mel spectrogram.
    Allows decoupling librosa dependency; saved using:
        np.savez_compressed(
            "mel_filters.npz",
            mel_80=librosa.filters.mel(sr=16000, n_fft=400, n_mels=80),
        )
    """
    assert n_mels == 80, f"Unsupported n_mels: {n_mels}"
    with np.load(os.path.join(os.path.dirname(__file__), "mel_filters.npz")) as f:
        return torch.from_numpy(f[f"mel_{n_mels}"]).to(device)

class Wav2Vec(nn.Module):
    def __init__(self, is_attack=False, requires_grad=False):
        super(Wav2Vec, self).__init__()
        
        # First we take the pretrained xlsr model
        self.backbone_model = Wav2Vec2Model.from_pretrained(
            "facebook/wav2vec2-base-960h",
            output_hidden_states=True
        )
        for name, param in self.backbone_model.named_parameters(): param.requires_grad = requires_grad
        
    def forward(self, x, norm="nonorm", length=None, is_attack=False):
        # 1. feature extraction and projections
        if not is_attack:
            with torch.no_grad():
                x = self.backbone_model.feature_extractor(x)
                x = x.transpose(1, 2) # New version of huggingface
                x, _ = self.backbone_model.feature_projection(x) # New version of huggingface
        else:
            self.backbone_model.feature_extractor.training = False
            x = self.backbone_model.feature_extractor(x)
            x = x.transpose(1, 2) # New version of huggingface
            x, _ = self.backbone_model.feature_projection(x) # New version of huggingface
        # 2. get length and mask
        mask = None
        if length is not None:
            length = self.get_feat_extract_output_lengths(length.detach().cpu())
            mask = prepare_mask(length, x.shape[:2], x.dtype)
            length, mask = length.cuda(), mask.cuda()
        # 3. transformer encoding features
        if not is_attack:
            with torch.no_grad():
                feat = self.backbone_model.encoder(
                    x, attention_mask=mask, 
                    output_hidden_states=True
                ).hidden_states
        else:
            feat = self.backbone_model.encoder(
                x, attention_mask=mask, 
                output_hidden_states=True
            ).hidden_states

        # 4. stacked feature
        stacked_feature = torch.stack(feat, dim=0)[1:]
        # 5. return feature, length and masks
        if length is not None: return stacked_feature, length, mask
        return stacked_feature
    
    # From huggingface
    def get_feat_extract_output_lengths(self, input_length):
        """
        Computes the output length of the convolutional layers
        """
        def _conv_out_length(input_length, kernel_size, stride):
            # 1D convolutional layer output length formula taken
            # from https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
            return (input_length - kernel_size) // stride + 1
        for kernel_size, stride in zip(self.backbone_model.config.conv_kernel, self.backbone_model.config.conv_stride):
            input_length = _conv_out_length(input_length, kernel_size, stride)
        return input_length

class APC(nn.Module):
    def __init__(self):
        super(APC, self).__init__()
        # First we take the apc model
        self.backbone_model = getattr(hub, "apc")()
        # setting require grad = true only if we want to fine tune the pretrained model
        for name, param in self.backbone_model.named_parameters(): param.requires_grad = False
        
    def forward(self, x, norm="nonorm", length=None, is_attack=False):
        # 1. if input is train with variable length
        new_x = list()
        if length is not None:
             for idx in range(len(length)):
                new_x.append(x[idx][:length[idx]])
        # 2. infer hidden states
        # Benign case
        if not is_attack:
            with torch.no_grad():
                if length is not None: feat = self.backbone_model(new_x)['hidden_states']
                else: feat = self.backbone_model(x)['hidden_states']
        # Attack case
        else:
            if length is not None: feat = self.backbone_model(new_x)['hidden_states']
            else: feat = self.backbone_model(x)['hidden_states']
        # 3. stacked feature
        stacked_feature = torch.stack(feat, dim=0)
        # 4. get length and feature
        if length is not None:
            length = self.get_feat_extract_output_lengths(length.detach().cpu())
            mask = prepare_mask(length, feat[0].shape[:2], x.dtype)
            length, mask = length.cuda(), mask.cuda()
            return stacked_feature, length, mask
        return stacked_feature
    
    # From huggingface
    def get_feat_extract_output_lengths(self, input_length):
        """
        Computes the output length
        """
        def _out_length(input_length, window, stride):
            return (input_length - window) // stride + 1
        input_length = _out_length(input_length, 400, 160)
        return input_length

class TERA(nn.Module):
    def __init__(self):
        super(TERA, self).__init__()
        # First we take the apc model
        self.backbone_model = getattr(hub, "tera")()
        # setting require grad = true only if we want to fine tune the pretrained model
        for name, param in self.backbone_model.named_parameters(): param.requires_grad = False
        
    def forward(self, x, norm="nonorm", length=None, is_attack=False):
        # 1. if input is train with variable length
        new_x = list()
        if length is not None:
             for idx in range(len(length)):
                new_x.append(x[idx][:length[idx]])
        # 2. infer hidden states
        # Benign case
        if not is_attack:
            with torch.no_grad():
                if length is not None: feat = self.backbone_model(new_x)['hidden_states']
                else: feat = self.backbone_model(x)['hidden_states']
        # Attack case
        else:
            if length is not None: feat = self.backbone_model(new_x)['hidden_states']
            else: feat = self.backbone_model(x)['hidden_states']
        # 3. stacked feature
        stacked_feature = torch.stack(feat, dim=0)
        # 4. get length and feature
        if length is not None:
            length = self.get_feat_extract_output_lengths(length.detach().cpu())
            mask = prepare_mask(length, feat[0].shape[:2], x.dtype)
            length, mask = length.cuda(), mask.cuda()
            return stacked_feature, length, mask
        return stacked_feature
    
    # From huggingface
    def get_feat_extract_output_lengths(self, input_length):
        """
        Computes the output length
        """
        def _out_length(input_length, window, stride):
            return (input_length - window) // stride + 1
        input_length = _out_length(input_length, 400, 160)
        return input_length


class WavLM(nn.Module):
    def __init__(self, is_attack=False, finetune="frozen"):
        super(WavLM, self).__init__()
        
        # First we take the pretrained xlsr model
        self.backbone_model = WavLMModel.from_pretrained(
            "microsoft/wavlm-base-plus",
            output_hidden_states=True
        )
        self.finetune = finetune
        # setting require grad = true only if we want to fine tune the pretrained model
        if is_attack:
            for name, param in self.backbone_model.named_parameters(): param.requires_grad = True
        else:
            if finetune == "frozen": 
                for name, param in self.backbone_model.named_parameters(): param.requires_grad = False
            elif finetune == "unfrozen_last_layer":
                for name, param in self.backbone_model.named_parameters(): 
                    if "layers.11" in name: param.requires_grad = True
                    else: param.requires_grad = False
            elif finetune == "unfrozen":
                for name, param in self.backbone_model.encoder.named_parameters(): 
                    param.requires_grad = True
        
    def forward(self, x, norm="nonorm", length=None, is_attack=False):
        # 1. Feature extraction and projections
        if not is_attack:
            with torch.no_grad():
                x = self.backbone_model.feature_extractor(x)
                x = x.transpose(1, 2) # New version of huggingface
                x, _ = self.backbone_model.feature_projection(x) # New version of huggingface
        else:
            # setting require grad = true only if we want to fine tune the pretrained model
            self.backbone_model.feature_extractor.training = False
            x = self.backbone_model.feature_extractor(x)
            x = x.transpose(1, 2) # New version of huggingface
            x, _ = self.backbone_model.feature_projection(x) # New version of huggingface

        # 2. get length and mask
        mask = None
        if length is not None:
            length = self.get_feat_extract_output_lengths(length.detach().cpu())
            mask = prepare_mask(length, x.shape[:2], x.dtype)
            length, mask = length.cuda(), mask.cuda()
        # 3. transformer encoding features
        if not is_attack:
            if self.finetune == "frozen": 
                with torch.no_grad():
                    feat = self.backbone_model.encoder(
                        x, attention_mask=mask, output_hidden_states=True
                    ).hidden_states
            else:
                feat = self.backbone_model.encoder(
                    x, attention_mask=mask, output_hidden_states=True
                ).hidden_states
        else:
            feat = self.backbone_model.encoder(
                x, attention_mask=mask, output_hidden_states=True
            ).hidden_states
        
        # 4. stacked feature
        stacked_feature = torch.stack(feat, dim=0)[1:]
        # 5. return feature, length and masks
        if length is not None: return stacked_feature, length, mask
        return stacked_feature
    
    # From huggingface
    def get_feat_extract_output_lengths(self, input_length):
        """
        Computes the output length of the convolutional layers
        """
        def _conv_out_length(input_length, kernel_size, stride):
            # 1D convolutional layer output length formula taken
            # from https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
            return (input_length - kernel_size) // stride + 1
        for kernel_size, stride in zip(self.backbone_model.config.conv_kernel, self.backbone_model.config.conv_stride):
            input_length = _conv_out_length(input_length, kernel_size, stride)
        return input_length


class WhisperTiny(nn.Module):
    def __init__(self, is_attack=False, finetune="frozen"):
        super(WhisperTiny, self).__init__()
        
        # First we take the pretrained model
        self.backbone_model = WhisperModel.from_pretrained("openai/whisper-tiny")
        self.feature_extractor = AutoFeatureExtractor.from_pretrained("openai/whisper-tiny")
        self.embed_positions = copy.deepcopy(self.backbone_model.encoder.embed_positions.weight)
        self.finetune = finetune
        
        # setting require grad = true only if we want to fine tune the pretrained model
        # if not is_attack:
        if finetune == "frozen": 
            for name, param in self.backbone_model.named_parameters(): param.requires_grad = False
        elif finetune == "unfrozen_last_layer":
            # We train the transformer but not the positional embeddings and conv layers
            for name, param in self.backbone_model.encoder.named_parameters(): 
                if "layers.3" in name: param.requires_grad = True
                else: param.requires_grad = False
            self.backbone_model.encoder.embed_positions.requires_grad = False
        elif finetune == "unfrozen":
            # We train the transformer but not the positional embeddings and conv layers
            for name, param in self.backbone_model.encoder.named_parameters(): 
                if "encoder.conv" in name: param.requires_grad = False
                else: param.requires_grad = True
            self.backbone_model.encoder.embed_positions.requires_grad = False
        
    def forward(self, x, input_features=None, norm="nonorm", length=None, is_attack=False):
        
        # Get the max audio length in a batch
        if length is not None: max_audio_len = length.max().detach().cpu()
        mask = None
        
        # 1. Feature extraction
        if length is not None:
            # Append to list for feature_extractor to work
            new_x = list()
            for idx in range(len(length)):
                new_x.append(x[idx].detach().cpu().numpy())
            
            # Max length is max audio len in a batch
            input_features = self.feature_extractor(
                new_x,
                return_tensors="pt", 
                sampling_rate=16000,
                max_length=max_audio_len
            )
            
            # Max length is max audio len in a batch
            # Return length
            length = self.get_feat_extract_output_lengths(length.detach().cpu())
            max_len = length.max()

            # Replace positional embeddings
            self.backbone_model.encoder.embed_positions = self.backbone_model.encoder.embed_positions.from_pretrained(self.embed_positions[:max_len])
            
            if self.finetune == "frozen": 
                with torch.no_grad():
                    feat = self.backbone_model.encoder(
                        input_features.input_features.cuda(), 
                        output_hidden_states=True
                    ).hidden_states
            else:
                feat = self.backbone_model.encoder(
                    input_features.input_features.cuda(),
                    output_hidden_states=True
                ).hidden_states
            mask = prepare_mask(length, feat[0].shape[:2], x.dtype)
            length, mask = length.cuda(), mask.cuda()

            # Stacked feature
            stacked_feature = torch.stack(feat, dim=0)[1:]
        else:
            if is_attack:
                # The preprocessing was referenced from Whisper
                # https://github.com/openai/whisper/blob/main/whisper/audio.py
                # We use whisper implementation as it allows differentiations of DFT
                audio = x[0].cpu().unsqueeze(dim=0)
                window = torch.hann_window(400).to(audio.device)
                stft = torch.stft(audio, 400, 160, window=window, return_complex=True)
                magnitudes = stft[..., :-1].abs() ** 2

                filters = mel_filters(audio.device, 80)
                mel_spec = filters @ magnitudes

                log_spec = torch.clamp(mel_spec, min=1e-10).log10()
                log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
                input_features = (log_spec + 4.0) / 4.0
                input_features = input_features.cuda()
            
            if input_features is None:
                input_features = self.feature_extractor(
                    x[0].detach().cpu(), 
                    return_tensors="pt", 
                    sampling_rate=16000,
                    max_length=len(x[0])
                )
                input_features = input_features.input_features.cuda()
            
            # Return length
            length = self.get_feat_extract_output_lengths(len(x[0]))
            
            # Replace positional embeddings
            self.backbone_model.encoder.embed_positions = self.backbone_model.encoder.embed_positions.from_pretrained(self.embed_positions[:length])
            if not is_attack:
                with torch.no_grad():
                    # Output shape: N x 1 x T x D
                    feat = self.backbone_model.encoder(
                        input_features, 
                        output_hidden_states=True
                    ).hidden_states
            else:
                # input_features.requires_grad = True
                # Output shape: N x 1 x T x D
                feat = self.backbone_model.encoder(
                    input_features,
                    output_hidden_states=True
                ).hidden_states
            stacked_feature = torch.stack(feat, dim=0)[1:]
        # 5. return feature, length and masks
        if mask is not None: return stacked_feature, length, mask
        return stacked_feature
    
    # From huggingface
    def get_feat_extract_output_lengths(self, input_lengths):
        """
        Computes the output length of the convolutional layers
        """
        input_lengths = input_lengths // 160
        input_lengths = (input_lengths - 1) // 2 + 1
        return input_lengths

class WhisperBase(nn.Module):
    def __init__(self, is_attack=False):
        super(WhisperBase, self).__init__()
        
        # First we take the pretrained model
        self.backbone_model = WhisperModel.from_pretrained("openai/whisper-base")
        self.feature_extractor = AutoFeatureExtractor.from_pretrained("openai/whisper-base")
        self.embed_positions = copy.deepcopy(self.backbone_model.encoder.embed_positions.weight)

        # setting require grad = true only if we want to fine tune the pretrained model
        for name, param in self.backbone_model.named_parameters(): param.requires_grad = False
        
    def forward(self, x, norm="nonorm", length=None, is_attack=False):
        
        # Get the max audio length in a batch
        if length is not None: max_audio_len = length.max().detach().cpu()
        mask = None
        
        # 1. Feature extraction
        if length is not None:
            # Append to list for feature_extractor to work
            new_x = list()
            for idx in range(len(length)):
                new_x.append(x[idx].detach().cpu().numpy())
            # Max length is max audio len in a batch
            input_features = self.feature_extractor(
                new_x,
                return_tensors="pt", 
                sampling_rate=16000,
                max_length=max_audio_len
            )
            
            # Return length
            length = self.get_feat_extract_output_lengths(length.detach().cpu())
            max_len = length.max()

            # Replace positional embeddings
            self.backbone_model.encoder.embed_positions = self.backbone_model.encoder.embed_positions.from_pretrained(self.embed_positions[:max_len])
            
            with torch.no_grad():
                feat = self.backbone_model.encoder(
                    input_features.input_features.cuda(),
                    output_hidden_states=True
                ).hidden_states
            mask = prepare_mask(length, feat[0].shape[:2], x.dtype)
            length, mask = length.cuda(), mask.cuda()

            # Stacked feature
            stacked_feature = torch.stack(feat, dim=0)[1:]
        else:
            if is_attack:
                # The preprocessing was referenced from Whisper
                # https://github.com/openai/whisper/blob/main/whisper/audio.py
                # We use whisper implementation as it allows differentiations of DFT
                audio = x[0].cpu().unsqueeze(dim=0)
                window = torch.hann_window(400).to(audio.device)
                stft = torch.stft(audio, 400, 160, window=window, return_complex=True)
                magnitudes = stft[..., :-1].abs() ** 2

                filters = mel_filters(audio.device, 80)
                mel_spec = filters @ magnitudes

                log_spec = torch.clamp(mel_spec, min=1e-10).log10()
                log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
                input_features = (log_spec + 4.0) / 4.0
                input_features = input_features.cuda()
            else:
                input_features = self.feature_extractor(
                    x[0].detach().cpu(), 
                    return_tensors="pt", 
                    sampling_rate=16000,
                    max_length=len(x[0])
                )
                input_features = input_features.input_features.cuda()
            
            # Return length
            length = self.get_feat_extract_output_lengths(len(x[0]))
            
            # Replace positional embeddings
            self.backbone_model.encoder.embed_positions = self.backbone_model.encoder.embed_positions.from_pretrained(self.embed_positions[:length])
            if not is_attack:
                with torch.no_grad():
                    # Output shape: N x 1 x T x D
                    feat = self.backbone_model.encoder(
                        input_features,
                        output_hidden_states=True
                    ).hidden_states
            else:
                # Output shape: N x 1 x T x D
                feat = self.backbone_model.encoder(
                    input_features,
                    output_hidden_states=True
                ).hidden_states
            stacked_feature = torch.stack(feat, dim=0)[1:]
        # 5. return feature, length and masks
        if mask is not None: return stacked_feature, length, mask
        return stacked_feature
    
    # From huggingface
    def get_feat_extract_output_lengths(self, input_lengths):
        """
        Computes the output length of the convolutional layers
        """
        input_lengths = input_lengths // 160
        input_lengths = (input_lengths - 1) // 2 + 1
        return input_lengths


class WhisperSmall(nn.Module):
    def __init__(self, is_attack=False):
        super(WhisperSmall, self).__init__()
        
        # First we take the pretrained model
        self.backbone_model = WhisperModel.from_pretrained("openai/whisper-small")
        self.feature_extractor = AutoFeatureExtractor.from_pretrained("openai/whisper-small")
        self.embed_positions = copy.deepcopy(self.backbone_model.encoder.embed_positions.weight)

        # setting require grad = true only if we want to fine tune the pretrained model
        for name, param in self.backbone_model.named_parameters(): param.requires_grad = False
        
    def forward(self, x, norm="nonorm", length=None, is_attack=False):
        
        # Get the max audio length in a batch
        if length is not None: max_audio_len = length.max().detach().cpu()
        mask = None
        
        # 1. Feature extraction
        if length is not None:
            # Append to list for feature_extractor to work
            new_x = list()
            for idx in range(len(length)):
                new_x.append(x[idx].detach().cpu().numpy())
            # Max length is max audio len in a batch
            input_features = self.feature_extractor(
                new_x,
                return_tensors="pt", 
                sampling_rate=16000,
                max_length=max_audio_len
            )
            
            # Return length
            length = self.get_feat_extract_output_lengths(length.detach().cpu())
            max_len = length.max()

            # Replace positional embeddings
            self.backbone_model.encoder.embed_positions = self.backbone_model.encoder.embed_positions.from_pretrained(self.embed_positions[:max_len])
            
            with torch.no_grad():
                feat = self.backbone_model.encoder(
                    input_features.input_features.cuda(),
                    output_hidden_states=True
                ).hidden_states
            mask = prepare_mask(length, feat[0].shape[:2], x.dtype)
            length, mask = length.cuda(), mask.cuda()

            # Stacked feature
            stacked_feature = torch.stack(feat, dim=0)[1:]
        else:
            if is_attack:
                # The preprocessing was referenced from Whisper
                # https://github.com/openai/whisper/blob/main/whisper/audio.py
                # We use whisper implementation as it allows differentiations of DFT
                audio = x[0].cpu().unsqueeze(dim=0)
                window = torch.hann_window(400).to(audio.device)
                stft = torch.stft(audio, 400, 160, window=window, return_complex=True)
                magnitudes = stft[..., :-1].abs() ** 2

                filters = mel_filters(audio.device, 80)
                mel_spec = filters @ magnitudes

                log_spec = torch.clamp(mel_spec, min=1e-10).log10()
                log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
                input_features = (log_spec + 4.0) / 4.0
                input_features = input_features.cuda()
            else:
                input_features = self.feature_extractor(
                    x[0].detach().cpu(), 
                    return_tensors="pt", 
                    sampling_rate=16000,
                    max_length=len(x[0])
                )
                input_features = input_features.input_features.cuda()
            
            # Return length
            length = self.get_feat_extract_output_lengths(len(x[0]))
            
            # Replace positional embeddings
            self.backbone_model.encoder.embed_positions = self.backbone_model.encoder.embed_positions.from_pretrained(self.embed_positions[:length])
            if not is_attack:
                with torch.no_grad():
                    # Output shape: N x 1 x T x D
                    feat = self.backbone_model.encoder(
                        input_features,
                        output_hidden_states=True
                    ).hidden_states
            else:
                # Output shape: N x 1 x T x D
                feat = self.backbone_model.encoder(
                    input_features,
                    output_hidden_states=True
                ).hidden_states

            stacked_feature = torch.stack(feat, dim=0)[1:]
        # 5. return feature, length and masks
        if mask is not None: return stacked_feature, length, mask
        return stacked_feature
    
    # From huggingface
    def get_feat_extract_output_lengths(self, input_lengths):
        """
        Computes the output length of the convolutional layers
        """
        input_lengths = input_lengths // 160
        input_lengths = (input_lengths - 1) // 2 + 1
        return input_lengths


def prepare_mask(length, shape, dtype):
    # Modified from huggingface
    mask = torch.zeros(
        shape, dtype=dtype
    )
    # these two operations makes sure that all values
    # before the output lengths indices are attended to
    mask[(torch.arange(mask.shape[0]), length.cpu() - 1)] = 1
    mask = mask.flip([-1]).cumsum(-1).flip([-1]).bool()
    return mask
    