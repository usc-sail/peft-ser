# part of the code was referenced from SUPERB: https://github.com/s3prl/s3prl
# and https://github.com/wngh1187/IPET/blob/main/Speechcommands_V2/W2V2/models/W2V2.py
import pdb
import copy
import torch
import argparse
import loralib as lora
import transformers.models.whisper.modeling_whisper as whisper

from torch import nn
from .adapter import Adapter
from torch.nn import functional as F
from transformers.activations import ACT2FN
from transformers import WhisperModel, AutoFeatureExtractor

model_config = {
    "whisper_base_lora_16_conv_output": {
        "pretrained_model": "whisper_base",
        "finetune_method": "lora",
        "lora_rank": 16,
        "use_conv_outpu": True,
        "classifier_link": "https://www.dropbox.com/scl/fi/sq6w9vfrlyd0t3y5xwnnm/whisper_base_lora_16_conv_output.pt?rlkey=xtot2gi9au83xi4f3kxzwjac9&dl=0",
        "lora_link": "https://www.dropbox.com/scl/fi/46bb08ek2zbt32fmqrg21/whisper_base_lora_16_conv_output_lora.pt?rlkey=452zdg99to7ntol3rjrtgccog&dl=0"
    },
    "whisper_base_lora_8_conv_output": {
        "pretrained_model": "whisper_base",
        "finetune_method": "lora",
        "lora_rank": 8,
        "use_conv_outpu": True,
        "classifier_link": "https://www.dropbox.com/scl/fi/sq6w9vfrlyd0t3y5xwnnm/whisper_base_lora_8_conv_output.pt?rlkey=xtot2gi9au83xi4f3kxzwjac9&dl=0",
        "lora_link": "https://www.dropbox.com/scl/fi/46bb08ek2zbt32fmqrg21/whisper_base_lora_8_conv_output_lora.pt?rlkey=452zdg99to7ntol3rjrtgccog&dl=0"
    },
    "whisper_small_lora_16_conv_output": {
        "pretrained_model": "whisper_small",
        "finetune_method": "lora",
        "lora_rank": 16,
        "use_conv_outpu": True,
        "classifier_link": "https://www.dropbox.com/scl/fi/sq6w9vfrlyd0t3y5xwnnm/whisper_small_lora_16_conv_output.pt?rlkey=xtot2gi9au83xi4f3kxzwjac9&dl=0",
        "lora_link": "https://www.dropbox.com/scl/fi/46bb08ek2zbt32fmqrg21/whisper_small_lora_16_conv_output_lora.pt?rlkey=452zdg99to7ntol3rjrtgccog&dl=0"
    },
    "whisper_small_lora_8_conv_output": {
        "pretrained_model": "whisper_small",
        "finetune_method": "lora",
        "lora_rank": 8,
        "use_conv_outpu": True,
        "classifier_link": "https://www.dropbox.com/scl/fi/sq6w9vfrlyd0t3y5xwnnm/whisper_small_lora_8_conv_output.pt?rlkey=xtot2gi9au83xi4f3kxzwjac9&dl=0",
        "lora_link": "https://www.dropbox.com/scl/fi/46bb08ek2zbt32fmqrg21/whisper_small_lora_8_conv_output_lora.pt?rlkey=452zdg99to7ntol3rjrtgccog&dl=0"
    },
    "whisper_tiny_lora_16_conv_output": {
        "pretrained_model": "whisper_tiny",
        "finetune_method": "lora",
        "lora_rank": 16,
        "use_conv_outpu": True,
        "classifier_link": "https://www.dropbox.com/scl/fi/sq6w9vfrlyd0t3y5xwnnm/whisper_tiny_lora_16_conv_output.pt?rlkey=xtot2gi9au83xi4f3kxzwjac9&dl=0",
        "lora_link": "https://www.dropbox.com/scl/fi/46bb08ek2zbt32fmqrg21/whisper_tiny_lora_16_conv_output_lora.pt?rlkey=452zdg99to7ntol3rjrtgccog&dl=0"
    },
    "whisper_tiny_lora_8_conv_output": {
        "pretrained_model": "whisper_tiny",
        "finetune_method": "lora",
        "lora_rank": 8,
        "use_conv_outpu": True,
        "classifier_link": "https://www.dropbox.com/scl/fi/sq6w9vfrlyd0t3y5xwnnm/whisper_tiny_lora_8_conv_output.pt?rlkey=xtot2gi9au83xi4f3kxzwjac9&dl=0",
        "lora_link": "https://www.dropbox.com/scl/fi/46bb08ek2zbt32fmqrg21/whisper_tiny_lora_8_conv_output_lora.pt?rlkey=452zdg99to7ntol3rjrtgccog&dl=0"
    }
}

# Attach code is from huggingface
class WhisperEncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.d_model
        self.self_attn = whisper.WhisperAttention(
            embed_dim=self.embed_dim,
            num_heads=config.encoder_attention_heads,
            dropout=config.attention_dropout,
        )
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)
        self.config = config
        
        if self.config.finetune_method == "embedding_prompt":
            self.embed_prompt = nn.Parameter(torch.randn([1, self.config.embedding_prompt_dim, self.embed_dim]))
            nn.init.xavier_uniform_(self.embed_prompt)
        if self.config.finetune_method == "lora" or self.config.finetune_method == "combined":
            self.fc1 = lora.Linear(self.embed_dim, config.encoder_ffn_dim, r=config.lora_rank)
            self.fc2 = lora.Linear(config.encoder_ffn_dim, self.embed_dim, r=config.lora_rank)
            
        if self.config.finetune_method == "adapter" or self.config.finetune_method == "adapter_l" or self.config.finetune_method == "combined":
            self.adapter = Adapter(
                config, 
                d_model=self.embed_dim,
                dropout=0.1, 
                bottleneck=config.adapter_hidden_dim, 
                adapter_scalar=0.1
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        layer_head_mask: torch.Tensor,
        output_attentions: bool = False,
    ):
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(seq_len, batch, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (`torch.FloatTensor`): mask for attention heads in a given layer of size
                `(encoder_attention_heads,)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        if self.config.finetune_method == "embedding_prompt" or self.config.finetune_method == "combined":
            hidden_states = torch.cat((self.embed_prompt.repeat(hidden_states.size(0), 1, 1), hidden_states), dim=1)
        
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states, attn_weights, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        residual = hidden_states
        
        # Adapter
        if self.config.finetune_method == "adapter":
            adapt_h = self.adapter(hidden_states)
        
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        
        # Adapter
        if self.config.finetune_method == "adapter_l" or self.config.finetune_method == "combined": 
            hidden_states = hidden_states + self.adapter(hidden_states)
        
        hidden_states = residual + hidden_states
        
        if hidden_states.dtype == torch.float16 and (
            torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any()
        ):
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        
        # Adapter
        if self.config.finetune_method == "adapter": 
            hidden_states = hidden_states + adapt_h
        
        if self.config.finetune_method == "embedding_prompt" or self.config.finetune_method == "combined":
            hidden_states = hidden_states[:, self.config.embedding_prompt_dim:, :]
        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs
   
class WhisperSER(nn.Module):
    def __init__(
        self, 
        pretrained_model:       str = "whisper_tiny",
        hidden_dim:             int = 256,
        output_class_num:       int = 4,
        finetune_method:        str = "lora",
        adapter_hidden_dim:     int = 128,
        embedding_prompt_dim:   int = 3,
        lora_rank:              int = 8,
        cache_dir:              str = ".",
        use_conv_output:        bool = True,
        enable_peft_training:   bool = True
    ):
        super(WhisperSER, self).__init__()
        # 1. We Load the model first with weights
        self.feature_extractor = AutoFeatureExtractor.from_pretrained("openai/whisper-tiny")
        self.pretrained_model = pretrained_model
        
        if self.pretrained_model == "whisper_tiny":
            self.backbone_model = WhisperModel.from_pretrained(
                "openai/whisper-tiny",
                output_hidden_states=True,
                cache_dir=cache_dir
            )
        elif self.pretrained_model == "whisper_base":
            self.backbone_model = WhisperModel.from_pretrained(
                "openai/whisper-base",
                output_hidden_states=True,
                cache_dir=cache_dir
            )
        elif self.pretrained_model == "whisper_small":
            self.backbone_model = WhisperModel.from_pretrained(
                "openai/whisper-small",
                output_hidden_states=True,
                cache_dir=cache_dir
            )
        elif self.pretrained_model == "whisper_medium":
            self.backbone_model = WhisperModel.from_pretrained(
                "openai/whisper-medium",
                output_hidden_states=True,
                cache_dir=cache_dir
            )
        self.embed_positions = copy.deepcopy(self.backbone_model.encoder.embed_positions.weight)
        self.embed_positions.requires_grad = False
        state_dict = self.backbone_model.state_dict()
        # 2. Read the model config
        self.model_config = self.backbone_model.config
        self.model_config.finetune_method        = finetune_method
        self.model_config.adapter_hidden_dim     = adapter_hidden_dim
        self.model_config.embedding_prompt_dim   = embedding_prompt_dim
        self.model_config.lora_rank              = lora_rank
        
        # 3. Config encoder layers with adapter or embedding prompt
        self.backbone_model.encoder.layers = nn.ModuleList(
            [WhisperEncoderLayer(self.model_config) for _ in range(self.model_config.encoder_layers)]
        )
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
        if self.use_conv_output:
            num_layers = self.model_config.num_hidden_layers + 1  # transformer layers + input embeddings
            self.layer_weights = nn.Parameter(torch.ones(num_layers)/num_layers)
        else:
            num_layers = self.model_config.num_hidden_layers
            self.layer_weights = nn.Parameter(torch.zeros(num_layers))
        
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
        # 0. check input length
        assert len(x.shape) == 2, "Input data shape wrong"
        assert len(x[0]) <= 10 * 16000, "SER training was using 10s window frame, please crop your data to 10s"

        # 1. feature extraction and projections
        if length is not None:
            max_audio_len = length.max().detach().cpu()
            # Append to list for feature_extractor to work
            new_x = list()
            for idx in range(len(length)):
                new_x.append(x[idx].detach().cpu().numpy())
            
            # Max length is max audio len in a batch
            features = self.feature_extractor(
                new_x,
                return_tensors="pt", 
                sampling_rate=16000,
                max_length=max_audio_len
            )
            features = features.input_features.to(x.device)
        else:
            features = self.feature_extractor(
                x[0].detach().cpu(), 
                return_tensors="pt", 
                sampling_rate=16000,
                max_length=len(x[0])
            )
            features = features.input_features.to(x.device)
        
        # 2. get length and mask
        if length is not None:
            length = self.get_feat_extract_output_lengths(length.detach().cpu())
            max_len = length.max()
            # Replace positional embeddings
            self.backbone_model.encoder.embed_positions = self.backbone_model.encoder.embed_positions.from_pretrained(self.embed_positions[:max_len])
        else:
            tmp_length = self.get_feat_extract_output_lengths(len(x[0]))
            # Replace positional embeddings
            self.backbone_model.encoder.embed_positions = self.backbone_model.encoder.embed_positions.from_pretrained(self.embed_positions[:tmp_length])
            
        # 3. transformer encoding features
        features = self.backbone_model.encoder(
            features, output_hidden_states=True
        ).hidden_states
        
        # 4. stacked feature
        if self.use_conv_output:
            stacked_feature = torch.stack(features, dim=0)
        else:
            stacked_feature = torch.stack(features, dim=0)[1:]
        
        # 5. Weighted sum
        _, *origin_shape = stacked_feature.shape
        # Return transformer enc outputs [num_enc_layers, B, T, D]
        if self.use_conv_output:
            stacked_feature = stacked_feature.view(self.backbone_model.config.num_hidden_layers+1, -1)
        else:
            stacked_feature = stacked_feature.view(self.backbone_model.config.num_hidden_layers, -1)
        norm_weights = F.softmax(self.layer_weights, dim=-1)
        
        # Perform weighted average
        weighted_feature = (norm_weights.unsqueeze(-1) * stacked_feature).sum(dim=0)
        features = weighted_feature.view(*origin_shape)
        
        # 6. Pass the weighted average to point-wise 1D Conv
        # B x T x D
        features = features.transpose(1, 2)
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
        
        # 8. Output predictions
        # B x D
        predicted = self.out_layer(features)
        return predicted
        
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
    
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='emo2vec finetune experiments')
    parser.add_argument(
        '--pretrained_model', 
        default='whisper_tiny',
        type=str, 
        help='finetune method: whisper_tiny, whisper_base, whisper_small'
    )
    
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
        help='prompt dimension'
    )
    
    parser.add_argument(
        '--lora_rank', 
        default=16,
        type=int, 
        help='adapter dimension'
    )
    
    args = parser.parse_args()
    model = WhisperWrapper(args).cuda()
    data = torch.zeros([1, 16000]).cuda()
    output = model(data)
    print(output.shape)