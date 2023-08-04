# part of the code was referenced from SUPERB: https://github.com/s3prl/s3prl
# and https://github.com/wngh1187/IPET/blob/main/Speechcommands_V2/W2V2/models/W2V2.py
import os
import pdb
import torch
import argparse
import loralib as lora
import transformers.models.wav2vec2.modeling_wav2vec2 as w2v2

from torch import nn
from .adapter import Adapter
from torch.nn import functional as F
from transformers import Wav2Vec2Model, AutoProcessor


class Wav2Vec2EncoderLayerStableLayerNorm(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = w2v2.Wav2Vec2Attention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=False,
        )
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.feed_forward = w2v2.Wav2Vec2FeedForward(config)
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.config = config

        if getattr(config, "adapter_attn_dim", None) is not None:
            self.adapter_layer = w2v2.Wav2Vec2AttnAdapterLayer(config)
        else:
            self.adapter_layer = None
        
        if self.config.finetune_method == "lora" or self.config.finetune_method == "combined":
            self.feed_forward.intermediate_dense    = lora.Linear(config.hidden_size, config.intermediate_size, r=config.lora_rank)
            self.feed_forward.output_dense          = lora.Linear(config.intermediate_size, config.hidden_size, r=config.lora_rank)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask=None,
        output_attentions: bool = False,
    ):
        attn_residual = hidden_states
        hidden_states = self.layer_norm(hidden_states)
        hidden_states, attn_weights, _ = self.attention(
            hidden_states, attention_mask=attention_mask, output_attentions=output_attentions
        )
        hidden_states = self.dropout(hidden_states)
        hidden_states = attn_residual + hidden_states
        hidden_states = hidden_states + self.feed_forward(self.final_layer_norm(hidden_states))

        if self.adapter_layer is not None:
            hidden_states = hidden_states + self.adapter_layer(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs

class MMSSER(nn.Module):
    def __init__(
        self, 
        pretrained_model:       str = "mms-300m",
        hidden_dim:             int = 256,
        output_class_num:       int = 4,
        finetune_method:        str = "lora",
        adapter_hidden_dim:     int = 128,
        embedding_prompt_dim:   int = 3,
        lora_rank:              int = 8,
        cache_dir:              str = ".",
        use_conv_output:        bool = True,
        enable_peft_training:   bool = True,
    ):
        super(MMSSER, self).__init__()
        # 1. We Load the model first with weights
        self.processor = AutoProcessor.from_pretrained(
            "facebook/mms-1b-all", 
            target_lang="eng", 
            cache_dir=cache_dir
        )
        
        self.backbone_model = Wav2Vec2Model.from_pretrained(
            "facebook/mms-300m",
            output_hidden_states=True,
            cache_dir=cache_dir
        )
        state_dict = self.backbone_model.state_dict()
        # 2. Read the model config
        self.model_config = self.backbone_model.config
        self.model_config.finetune_method        = finetune_method
        self.model_config.adapter_hidden_dim     = adapter_hidden_dim
        self.model_config.embedding_prompt_dim   = embedding_prompt_dim
        self.model_config.lora_rank              = lora_rank
        self.layer_norm                          = nn.LayerNorm(self.model_config.hidden_size, eps=self.model_config.layer_norm_eps)
        
        # 3. Config encoder layers with adapter or embedding prompt
        # pdb.set_trace()
        self.backbone_model.encoder.layers = nn.ModuleList([Wav2Vec2EncoderLayerStableLayerNorm(self.model_config) for i in range(self.model_config.num_hidden_layers)])
        # 4. Load the weights back
        msg = self.backbone_model.load_state_dict(state_dict, strict=False)
        # 5. Freeze the weights
        for name, p in self.backbone_model.named_parameters():
            if name in msg.missing_keys and enable_peft_training: 
                p.requires_grad = True
            else: 
                p.requires_grad = False
        self.finetune_method = finetune_method
        
        # 6. Downstream models
        self.model_seq = nn.Sequential(
            nn.Conv1d(self.model_config.hidden_size, hidden_dim, 1, padding=0),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Conv1d(hidden_dim, hidden_dim, 1, padding=0),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Conv1d(hidden_dim, hidden_dim, 1, padding=0)
        )
        self.use_conv_output = use_conv_output
        if use_conv_output:
            num_layers = self.model_config.num_hidden_layers + 1  # transformer layers + input embeddings
        else:
            num_layers = self.model_config.num_hidden_layers
        self.layer_weights = nn.Parameter(torch.ones(num_layers)/num_layers)
        
        self.out_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_class_num),
        )
        
        # initiate model setting
        self.init_peft_model_setting()
        
    def init_peft_model_setting(self):
        self.model_setting = self.finetune_method
        if self.finetune_method == "lora":
            self.model_setting += f"_{self.model_config.lora_rank}"
        elif self.finetune_method == "adapter":
            self.model_setting += f"_{self.model_config.adapter_hidden_dim}"
        elif self.finetune_method == "embedding_prompt":
            self.model_setting += f"_{self.model_config.embedding_prompt_dim}"
        if self.use_conv_output:
            self.model_setting += "_conv_output"

    def forward(self, x, length=None):
        # 1. feature extraction and projections
        with torch.no_grad():
            x = self.processor(x, sampling_rate=16000, return_tensors="pt")["input_values"][0].to(x.device)
            attention_mask = torch.ones(x.shape, device=x.device, dtype=torch.long)
            x = self.backbone_model.feature_extractor(x)
            x = x.transpose(1, 2) # New version of huggingface
            x, _ = self.backbone_model.feature_projection(x) # New version of huggingface
        
        # 2. get length and mask
        if length is not None:
            for i in range(len(length)):
                attention_mask[i, length[i]:] = 0

            length = self._get_feat_extract_output_lengths(length.detach().cpu())
            length = length.to(x.device)
            
        # 3. transformer encoding features
        if length is not None:
            # compute reduced attention_mask corresponding to feature vectors
            attention_mask = self._get_feature_vector_attention_mask(
                x.shape[1], attention_mask, add_adapter=False
            )
            x = self.backbone_model.encoder(
                x, 
                attention_mask=attention_mask,
                output_hidden_states=True
            ).hidden_states
        else:
            x = self.backbone_model.encoder(
                x, output_hidden_states=True
            ).hidden_states

        # 4. stacked feature
        stacked_feature = torch.stack(x, dim=0)[:]

        # 5. Weighted sum
        # pdb.set_trace()
        _, *origin_shape = stacked_feature.shape
        # Return transformer enc outputs [num_enc_layers, B, T, D]
        if self.use_conv_output:
            stacked_feature = stacked_feature.view(self.backbone_model.config.num_hidden_layers+1, -1)
        else:
            stacked_feature = stacked_feature.view(self.backbone_model.config.num_hidden_layers, -1)
        norm_weights = F.softmax(self.layer_weights, dim=-1)
        
        # Perform weighted average
        weighted_feature = (norm_weights.unsqueeze(-1) * stacked_feature).sum(dim=0)
        weighted_feature = weighted_feature.view(*origin_shape)
        
        # 6. Pass the weighted average to point-wise 1D Conv
        # B x T x D
        features = weighted_feature.transpose(1, 2)
        features = self.model_seq(features)
        features = features.transpose(1, 2)
        
        # 7. Pooling
        if length is not None:
            length = length.to(x.device)
            masks = torch.arange(features.size(1)).expand(length.size(0), -1).to(x.device) < length.unsqueeze(1)
            masks = masks.float()
            features = (features * masks.unsqueeze(-1)).sum(1) / length.unsqueeze(1)
        else:
            features = torch.mean(features, dim=1)

        # 7. Output predictions
        # B x D
        predicted = self.out_layer(features)
        return predicted
        
    # From huggingface
    def _get_feat_extract_output_lengths(self, input_length):
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
    
    def _get_feature_vector_attention_mask(
        self, feature_vector_length: int, attention_mask: torch.LongTensor, add_adapter=None
    ):
        # Effectively attention_mask.sum(-1), but not inplace to be able to run
        # on inference mode.
        non_padded_lengths = attention_mask.cumsum(dim=-1)[:, -1]

        output_lengths = self._get_feat_extract_output_lengths(non_padded_lengths)
        output_lengths = output_lengths.to(torch.long)

        batch_size = attention_mask.shape[0]

        attention_mask = torch.zeros(
            (batch_size, feature_vector_length), dtype=attention_mask.dtype, device=attention_mask.device
        )
        # these two operations makes sure that all values before the output lengths idxs are attended to
        attention_mask[(torch.arange(attention_mask.shape[0], device=attention_mask.device), output_lengths - 1)] = 1
        attention_mask = attention_mask.flip([-1]).cumsum(-1).flip([-1]).bool()
        return attention_mask

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
    

def stft(x):
    return torch.stft(
        x, 
        n_fft=400, 
        hop_length=320, 
        win_length=400, 
        center=True, 
        normalized=True, 
        onesided=True,
        pad_mode='reflect',
        return_complex=True,
        window=torch.hamming_window(400).to(x.device)
    )

def istft(fea, length):
    return torch.istft(
        fea, 
        n_fft=400, 
        hop_length=320, 
        win_length=400, 
        center=True, 
        normalized=True, 
        onesided=True, 
        window=torch.hamming_window(400).to(fea.device),
        return_complex=False,
        length=length
    )
    

class BLSTM(nn.Module):
    def __init__(self,input_size,hidden_size,num_layers=1,dropout=0):
        super().__init__()
        self.blstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            batch_first=True, 
            num_layers=num_layers, 
            dropout=dropout, 
            bidirectional=True
        )

    def forward(self,x):
        # import pdb;pdb.set_trace()
        out, _ = self.blstm(x)
        out = out[:,:,:int(out.size(-1)/2)]+out[:,:,int(out.size(-1)/2):] 
        return out
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='emo2vec finetune experiments')
    parser.add_argument(
        '--finetune_method', 
        default='none',
        type=str, 
        help='finetune method: adapter, embedding prompt, input prompt'
    )
    
    parser.add_argument(
        '--adapter_hidden_dim', 
        default=128,
        type=int, 
        help='adapter dimension'
    )
    
    parser.add_argument(
        '--embedding_prompt_dim', 
        default=5,
        type=int, 
        help='adapter dimension'
    )

    parser.add_argument(
        '--lora_rank', 
        default=16,
        type=int, 
        help='lora rank'
    )

    parser.add_argument(
        '--include_layer', 
        default=0,
        type=int, 
        help='include layer'
    )

    parser.add_argument(
        '--downstream_model', 
        default="ecapa",
        type=str, 
        help='downstream models'
    )

    parser.add_argument(
        '--pretrain_model', 
        default="mms",
        type=str, 
        help='pretrained models'
    )
    
    args = parser.parse_args()
    model = MMSWrapper(args)
    # model.eval()
    data = torch.zeros([1, 16000])
    output = model(data)