
import pdb
import torch
import itertools

from torch import nn
from torch import Tensor
from collections import OrderedDict
from typing import Optional, Callable
from torch.nn import functional as F
from torch.nn.functional import normalize
from torch.nn.utils.rnn import pad_packed_sequence
from torch.nn.utils.rnn import pack_padded_sequence

# The implementation was modified from SUPERB: https://github.com/s3prl/s3prl
class SelfAttentionPooling(nn.Module):
    """
    Implementation of SelfAttentionPooling
    Original Paper: Self-Attention Encoding and Pooling for Speaker Recognition
    https://arxiv.org/pdf/2008.01077v1.pdf
    """
    def __init__(self, input_dim):
        super(SelfAttentionPooling, self).__init__()
        self.W = nn.Linear(input_dim, 1)
        self.softmax = nn.functional.softmax

    def forward(self, batch_rep, mask=None):
        """
            N: batch size, T: sequence length, H: Hidden dimension
            input:
                batch_rep : size (N, T, H)
            attention_weight:
                att_w : size (N, T, 1)
            return:
                utter_rep: size (N, H)
        """
        att_logits = self.W(batch_rep).squeeze(-1)
        if mask is not None:
            att_mask = torch.zeros(mask.shape).cuda()
            att_mask[mask==False] = -10e6
            att_logits = att_mask + att_logits
        att_w = self.softmax(att_logits, dim=-1).unsqueeze(-1)
        utter_rep = torch.sum(batch_rep * att_w, dim=1)
        return utter_rep


class CNNSelfAttention(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim=256,
        kernel_size=1,
        padding=0,
        pooling=5,
        dropout=0.2,
        output_class_num=4,
        conv_layer=3,
        num_enc_layers=3,
        pooling_method="att"
    ):
        super(CNNSelfAttention, self).__init__()
        
        # 1D point-wise convolution
        if conv_layer == 3:
            self.model_seq = nn.Sequential(
                nn.Conv1d(input_dim, hidden_dim, kernel_size, padding=padding),
                nn.ReLU(),
                nn.Dropout(p=dropout),
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=padding),
                nn.ReLU(),
                nn.Dropout(p=dropout),
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=padding),
            )
        else:
            self.model_seq = nn.Sequential(
                nn.Conv1d(input_dim, hidden_dim, kernel_size, padding=padding),
                nn.ReLU(),
                nn.Dropout(p=dropout),
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=padding)
            )
        
        self.pooling_method = pooling_method
        self.num_enc_layers = num_enc_layers
        self.pooling = SelfAttentionPooling(hidden_dim)
        self.weights = nn.Parameter(torch.zeros(self.num_enc_layers))
        self.out_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_class_num),
        )
        self.init_weight()
    
    def init_weight(self):
        for m in self._modules:
            if type(m) == nn.Linear:
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                m.bias.data.fill_(0.01)
            if type(m) == nn.Conv1d:
                torch.nn.init.xavier_uniform(m.weight)
                m.bias.data.fill_(0.01)

        
    def forward(self, features, length=None, mask=None):
        # Weighted average of all layers outputs
        # Num_enc x B x T x D
        if len(features.shape) == 4: 
            _, *origin_shape = features.shape
            # Return transformer enc outputs [num_enc_layers, B, T, D]
            features = features.view(self.num_enc_layers, -1)
            norm_weights = F.softmax(self.weights, dim=-1)
            
            # Perform weighted average
            weighted_feature = (norm_weights.unsqueeze(-1) * features).sum(dim=0)
            features = weighted_feature.view(*origin_shape)
        
        # Pass the weighted average to point-wise 1D Conv
        # B x T x D
        features = features.transpose(1, 2)
        features = self.model_seq(features)
        features = features.transpose(1, 2)
        
        # Convolution and pooling
        # B x T x D
        if self.pooling_method == "att": features = self.pooling(features, mask).squeeze(-1)
        else: 
            if length is not None:
                masks = torch.arange(features.size(1)).expand(length.size(0), -1).cuda() < length.unsqueeze(1)
                masks = masks.float()
                features = (features * masks.unsqueeze(-1)).sum(1) / length.unsqueeze(1)
            else:
                features = torch.mean(features, dim=1)
        
        # Output predictions
        # B x D
        predicted = self.out_layer(features)
        return predicted


class DNNClassifier(nn.Module):
    def __init__(self, num_class):
        super(DNNClassifier, self).__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(768, 128),
            nn.ReLU(),
            nn.Linear(128, num_class),
        )
        
    def forward(self, x):
        # the features are like a spectrogram, an image with one channel
        feat = x.mean(dim=1)
        output = self.classifier(feat)
        return output
