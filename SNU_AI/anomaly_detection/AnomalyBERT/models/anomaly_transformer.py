"""
AnomalyBERT
################################################

Reference:
    Yungi Jeong et al. "AnomalyBERT: Self-Supervised Transformer for Time Series Anomaly Detection using Data Degradation Scheme" in ICLR Workshop, "Machine Learning for Internet of Things(IoT): Datasets, Perception, and Understanding" 2023.

Reference:
    https://github.com/Jhryu30/AnomalyBERT
"""

import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import numpy as np

import torch
import torch.nn as nn

from timm.models.layers import trunc_normal_

from models.transformer import get_transformer_encoder



# Anomaly Transformer
class AnomalyTransformer(nn.Module):
    def __init__(self, linear_embedding, transformer_encoder, mlp_layers, d_embed, patch_size, max_seq_len):
        """
        <class init args>
        linear_embedding : embedding layer to feed data into Transformer encoder
        transformer_encoder : Transformer encoder body
        mlp_layers : MLP layers to return output data
        d_embed : embedding dimension (in Transformer encoder)
        patch_size : number of data points for an embedded vector
        max_seq_len : maximum length of sequence (= window size)
        """
        super(AnomalyTransformer, self).__init__()
        self.linear_embedding = linear_embedding
        self.transformer_encoder = transformer_encoder
        self.mlp_layers = mlp_layers

        self.max_seq_len = max_seq_len
        self.patch_size = patch_size
        self.data_seq_len = patch_size * max_seq_len
    
    def forward(self, x):
        """
        <input info>
        x : (n_batch, n_token, d_data) = (_, max_seq_len*patch_size, _)
        """
        n_batch = x.shape[0]
        
        embedded_out = x.view(n_batch, self.max_seq_len, self.patch_size, -1).view(n_batch, self.max_seq_len, -1)
        embedded_out = self.linear_embedding(embedded_out)  # linear embedding
            
        transformer_out = self.transformer_encoder(embedded_out)  # Encode data.
        output = self.mlp_layers(transformer_out)  # Reconstruct data.
        return output.view(n_batch, self.max_seq_len, self.patch_size, -1).view(n_batch, self.data_seq_len, -1)
    
class Classification_AnomalyTransformer(nn.Module):
    def __init__(self, linear_embedding, transformer_encoder, mlp_layers4feature, mlp_layers4time, d_embed, patch_size, max_seq_len):
        """
        <class init args>
        linear_embedding : embedding layer to feed data into Transformer encoder
        transformer_encoder : Transformer encoder body
        mlp_layers : MLP layers to return output data
        d_embed : embedding dimension (in Transformer encoder)
        patch_size : number of data points for an embedded vector
        max_seq_len : maximum length of sequence (= window size)
        """
        super(Classification_AnomalyTransformer, self).__init__()
        self.linear_embedding = linear_embedding
        self.transformer_encoder = transformer_encoder
        self.mlp_layers4feature = mlp_layers4feature
        self.mlp_layers4time = mlp_layers4time

        self.max_seq_len = max_seq_len
        self.patch_size = patch_size
        self.data_seq_len = patch_size * max_seq_len
    
    def forward(self, x):
        """
        <input info>
        x : (n_batch, n_token, d_data) = (_, max_seq_len*patch_size, _)
        """
        n_batch = x.shape[0]
        
        embedded_out = x.view(n_batch, self.max_seq_len, self.patch_size, -1).view(n_batch, self.max_seq_len, -1)
        embedded_out = self.linear_embedding(embedded_out)  # linear embedding
            
        transformer_out = self.transformer_encoder(embedded_out)  # Encode data.
        output = self.mlp_layers4feature(transformer_out)  # Reconstruct data.
        output = self.mlp_layers4time(output.squeeze(-1))
        return output
    
    
# Get Anomaly Transformer.
def get_anomaly_transformer(input_d_data,
                            output_d_data,
                            patch_size,
                            d_embed=512,
                            hidden_dim_rate=4.,
                            max_seq_len=512,
                            positional_encoding=None,
                            relative_position_embedding=True,
                            transformer_n_layer=12,
                            transformer_n_head=8,
                            dropout=0.1):
    """
    <input info>
    input_d_data : data input dimension
    output_d_data : data output dimension
    patch_size : number of data points per embedded feature
    d_embed : embedding dimension (in Transformer encoder)
    hidden_dim_rate : hidden layer dimension rate to d_embed
    max_seq_len : maximum length of sequence (= window size)
    positional_encoding : positional encoding for embedded input; None/Sinusoidal/Absolute
    relative_position_embedding : relative position embedding option
    transformer_n_layer : number of Transformer encoder layers
    transformer_n_head : number of heads in multi-head attention module
    dropout : dropout rate
    """
    hidden_dim = int(hidden_dim_rate * d_embed)
    
    linear_embedding = nn.Linear(input_d_data*patch_size, d_embed)
    transformer_encoder = get_transformer_encoder(d_embed=d_embed,
                                                  positional_encoding=positional_encoding,
                                                  relative_position_embedding=relative_position_embedding,
                                                  n_layer=transformer_n_layer,
                                                  n_head=transformer_n_head,
                                                  d_ff=hidden_dim,
                                                  max_seq_len=max_seq_len,
                                                  dropout=dropout)
    mlp_layers = nn.Sequential(nn.Linear(d_embed, hidden_dim),
                               nn.GELU(),
                               nn.Linear(hidden_dim, output_d_data*patch_size))
    
    nn.init.xavier_uniform_(linear_embedding.weight)
    nn.init.zeros_(linear_embedding.bias)
    nn.init.xavier_uniform_(mlp_layers[0].weight)
    nn.init.zeros_(mlp_layers[0].bias)
    nn.init.xavier_uniform_(mlp_layers[2].weight)
    nn.init.zeros_(mlp_layers[2].bias)
    
    return AnomalyTransformer(linear_embedding,
                              transformer_encoder,
                              mlp_layers,
                              d_embed,
                              patch_size,
                              max_seq_len)

# Get Anomaly Transformer.
def get_classification_anomaly_transformer(input_d_data,
                            output_d_data,
                            patch_size,
                            d_embed=512,
                            hidden_dim_rate=4.,
                            max_seq_len=512,
                            positional_encoding=None,
                            relative_position_embedding=True,
                            transformer_n_layer=12,
                            transformer_n_head=8,
                            dropout=0.1):
    """
    <input info>
    input_d_data : data input dimension
    output_d_data : data output dimension
    patch_size : number of data points per embedded feature
    d_embed : embedding dimension (in Transformer encoder)
    hidden_dim_rate : hidden layer dimension rate to d_embed
    max_seq_len : maximum length of sequence (= window size)
    positional_encoding : positional encoding for embedded input; None/Sinusoidal/Absolute
    relative_position_embedding : relative position embedding option
    transformer_n_layer : number of Transformer encoder layers
    transformer_n_head : number of heads in multi-head attention module
    dropout : dropout rate
    """
    hidden_dim = int(hidden_dim_rate * d_embed)
    
    linear_embedding = nn.Linear(input_d_data*patch_size, d_embed)
    transformer_encoder = get_transformer_encoder(d_embed=d_embed,
                                                  positional_encoding=positional_encoding,
                                                  relative_position_embedding=relative_position_embedding,
                                                  n_layer=transformer_n_layer,
                                                  n_head=transformer_n_head,
                                                  d_ff=hidden_dim,
                                                  max_seq_len=max_seq_len,
                                                  dropout=dropout)
    mlp_layers4feature = nn.Sequential(nn.Linear(d_embed, d_embed//2),
                               nn.GELU(),
                               nn.Linear(d_embed//2, 1))
    
    mlp_layers4time = nn.Sequential(nn.Linear(d_embed, d_embed//2),
                               nn.GELU(),
                               nn.Linear(d_embed//2, output_d_data))
    
    nn.init.xavier_uniform_(linear_embedding.weight)
    nn.init.zeros_(linear_embedding.bias)
    nn.init.xavier_uniform_(mlp_layers4feature[0].weight)
    nn.init.zeros_(mlp_layers4feature[0].bias)
    nn.init.xavier_uniform_(mlp_layers4feature[2].weight)
    nn.init.zeros_(mlp_layers4feature[2].bias)
    nn.init.xavier_uniform_(mlp_layers4time[0].weight)
    nn.init.zeros_(mlp_layers4time[0].bias)
    nn.init.xavier_uniform_(mlp_layers4time[2].weight)
    nn.init.zeros_(mlp_layers4time[2].bias)
    
    return Classification_AnomalyTransformer(linear_embedding,
                              transformer_encoder,
                              mlp_layers4feature,
                              mlp_layers4time,
                              d_embed,
                              patch_size,
                              max_seq_len)