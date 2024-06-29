# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Changes have been made over the original file
# https://github.com/huggingface/pytorch-transformers/blob/v0.4.0/pytorch_pretrained_bert/modeling.py

"""PyTorch BERT model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import copy
import json
import math
import logging
import tarfile
import tempfile
import shutil
import numpy as np
import pickle

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
import torch.nn.functional as F

from .file_utils import cached_path
from .loss import LabelSmoothingLoss
from torch.nn.utils.rnn import pad_sequence
from .rank_loss import *

# import visdom

logger = logging.getLogger(__name__)
# vis = visdom.Visdom(port=8888, env='vlp')

# 默认用 bert-base-cased  编码器具有12个隐层, 输出768维张量, 12个自注意力头, 共110M参数量, 在不区分大小写的英文文本上进行训练而得到.
PRETRAINED_MODEL_ARCHIVE_MAP = {
    'bert-base-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased.tar.gz",
    'bert-large-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased.tar.gz",
    'bert-base-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased.tar.gz",
    'bert-large-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased.tar.gz",
    'bert-base-multilingual-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-uncased.tar.gz",
    'bert-base-multilingual-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-cased.tar.gz",
    # 'bert-base-chinese': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese.tar.gz",
    'bert-base-chinese': "/disk1/shipeng/slrBertTJUQA/bert-base-chinese",

    'bert-base-greek-uncased-v1': "/home/gaoliqing/shipeng/code/bert-base-greek-uncased-v1",
    'bert-base-german-cased-v1': "/home/gaoliqing/shipeng/code/bert-base-german-cased-v1"
}
CONFIG_NAME = 'bert_config.json'
WEIGHTS_NAME = 'pytorch_model.bin'


def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish}


class BertConfig(object):
    """Configuration class to store the configuration of a `BertModel`.
    """

    def __init__(self,
                 vocab_size_or_config_json_file,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=2,
                 relax_projection=0,
                 initializer_range=0.02,
                 task_idx=None,
                 fp32_embedding=False,
                 label_smoothing=None):
        """Constructs BertConfig.

        Args:
            vocab_size_or_config_json_file: Vocabulary size of `inputs_ids` in `BertModel`.
            hidden_size: Size of the encoder layers and the pooler layer.
            num_hidden_layers: Number of hidden layers in the Transformer encoder.
            num_attention_heads: Number of attention heads for each attention layer in
                the Transformer encoder.
            intermediate_size: The size of the "intermediate" (i.e., feed-forward)
                layer in the Transformer encoder.
            hidden_act: The non-linear activation function (function or string) in the
                encoder and pooler. If string, "gelu", "relu" and "swish" are supported.
            hidden_dropout_prob: The dropout probabilitiy for all fully connected
                layers in the embeddings, encoder, and pooler.
            attention_probs_dropout_prob: The dropout ratio for the attention
                probabilities.
            max_position_embeddings: The maximum sequence length that this model might
                ever be used with. Typically set this to something large just in case
                (e.g., 512 or 1024 or 2048).
            type_vocab_size: The vocabulary size of the `token_type_ids` passed into
                `BertModel`.
            initializer_range: The sttdev of the truncated_normal_initializer for
                initializing all weight matrices.
        """
        if isinstance(vocab_size_or_config_json_file, str):
            with open(vocab_size_or_config_json_file, "r", encoding='utf-8') as reader:
                json_config = json.loads(reader.read())
            for key, value in json_config.items():
                self.__dict__[key] = value
        elif isinstance(vocab_size_or_config_json_file, int):
            self.vocab_size = vocab_size_or_config_json_file
            self.hidden_size = hidden_size
            self.num_hidden_layers = num_hidden_layers
            self.num_attention_heads = num_attention_heads
            self.hidden_act = hidden_act
            self.intermediate_size = intermediate_size
            self.hidden_dropout_prob = hidden_dropout_prob
            self.attention_probs_dropout_prob = attention_probs_dropout_prob
            self.max_position_embeddings = max_position_embeddings
            self.type_vocab_size = type_vocab_size
            self.relax_projection = relax_projection
            self.initializer_range = initializer_range
            self.task_idx = task_idx
            self.fp32_embedding = fp32_embedding
            self.label_smoothing = label_smoothing
        else:
            raise ValueError("First argument must be either a vocabulary size (int)"
                             "or the path to a pretrained model config file (str)")

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `BertConfig` from a Python dictionary of parameters."""
        config = BertConfig(vocab_size_or_config_json_file=-1)
        for key, value in json_object.items():
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `BertConfig` from a json file of parameters."""
        with open(json_file, "r", encoding='utf-8') as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


try:
    from apex.normalization.fused_layer_norm import FusedLayerNorm as BertLayerNorm
except ImportError:
    print("Better speed can be achieved with apex installed from https://www.github.com/nvidia/apex.")


    class BertLayerNorm(nn.Module):
        def __init__(self, hidden_size, eps=1e-5):
            """Construct a layernorm module in the TF style (epsilon inside the square root).
            """
            super(BertLayerNorm, self).__init__()
            self.weight = nn.Parameter(torch.ones(hidden_size))
            self.bias = nn.Parameter(torch.zeros(hidden_size))
            self.variance_epsilon = eps

        def forward(self, x):
            u = x.mean(-1, keepdim=True)
            s = (x - u).pow(2).mean(-1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.variance_epsilon)
            return self.weight * x + self.bias


class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """

    def __init__(self, config):
        super(BertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(
            config.type_vocab_size, config.hidden_size)
        if hasattr(config, 'fp32_embedding'):
            self.fp32_embedding = config.fp32_embedding
        else:
            self.fp32_embedding = False

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-5)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, vis_feats, vis_pe, input_ids, token_type_ids=None, position_ids=None, vis_input=True,
                len_vis_input=49):
        if input_ids == None:
            words_embeddings = vis_feats
            position_embeddings = self.position_embeddings(vis_pe)
            if token_type_ids is None :
                token_type_ids = torch.zeros(vis_feats.size(0),vis_feats.size(1), dtype=torch.long ,device=vis_feats.device)
                token_type_ids.fill_(0)
            token_type_embeddings = self.token_type_embeddings(token_type_ids)
            # embeddings = words_embeddings + position_embeddings + token_type_embeddings
            embeddings = words_embeddings + position_embeddings
            if self.fp32_embedding:
                embeddings = embeddings.half()
            embeddings = self.LayerNorm(embeddings)
            embeddings = self.dropout(embeddings)
            return embeddings

        if vis_feats == None:
            text_seq_length = input_ids.size(1)
            if position_ids is None:
                position_ids = torch.arange(text_seq_length, dtype=torch.long, device=input_ids.device)
                position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
            if token_type_ids is None :
                token_type_ids = torch.zeros(input_ids.size(0),input_ids.size(1), dtype=torch.long ,device=input_ids.device)
                # 第一个[SEP]及之前为video，设为0，后面设为1
                token_type_ids.fill_(0)

            try:
                words_embeddings = self.word_embeddings(input_ids)
            except Exception as e:
                print(e)
                print(input_ids)
            # finally:
            #     print(input_ids)

            position_embeddings = self.position_embeddings(position_ids)
            # token_type_embeddings = self.token_type_embeddings(token_type_ids)
            embeddings = words_embeddings + position_embeddings
            if self.fp32_embedding:
                embeddings = embeddings.half()
            embeddings = self.LayerNorm(embeddings)
            embeddings = self.dropout(embeddings)
            return embeddings
    
        text_seq_length = input_ids.size(1)
        if position_ids is None:
            position_ids = torch.arange(
                input_ids.size(1) + vis_feats.size(1), dtype=torch.long, device=vis_feats.device)
            position_ids = position_ids.expand(input_ids.size(0),input_ids.size(1) + vis_feats.size(1))
            # position_ids = torch.arange(
            #     text_seq_length, dtype=torch.long, device=input_ids.device)
            # position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        # token_type_ids = torch.zeros(input_ids.size(0),input_ids.size(1) + vis_feats.size(1), dtype=torch.long ,device=input_ids.device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_ids.size(0),input_ids.size(1) + vis_feats.size(1), dtype=torch.long ,device=input_ids.device)
            # # 第一个[SEP]及之前为video，设为0，后面设为1
            # token_type_ids[:,2+vis_feats.size(1):].fill_(1)
            # 第一个[SEP]及之前为text，设为0，后面设为1
            token_type_ids[:,input_ids.size(1):].fill_(1)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        if vis_input and len_vis_input != 0:
            # words_embeddings = torch.cat((words_embeddings[:, :1], vis_feats,
            #                               words_embeddings[:, 1:]), dim=1)
            words_embeddings = torch.cat((words_embeddings, vis_feats,
                                          ), dim=1)
            # assert len_vis_input == 100, 'only support region attn!'

            # 文本和视频的embedding改为连续
            # vis_pe_embedding = self.position_embeddings(vis_pe)
            # position_embeddings = torch.cat((position_embeddings[:, :1], vis_pe_embedding,
            #                                  position_embeddings[:, 1:]), dim=1)  # hacky...
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        if self.fp32_embedding:
            embeddings = embeddings.half()
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertMultiScaleEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """

    def __init__(self, config):
        super(BertMultiScaleEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(
            config.type_vocab_size, config.hidden_size)
        if hasattr(config, 'fp32_embedding'):
            self.fp32_embedding = config.fp32_embedding
        else:
            self.fp32_embedding = False

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-5)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids):
        text_seq_length = input_ids.size(1)
        words_embeddings_fine = self.word_embeddings(input_ids)
        words_embeddings_coarse = torch.zeros(words_embeddings_fine.size(0), words_embeddings_fine.size(1) - 1, words_embeddings_fine.size(2),device = words_embeddings_fine.device)
        for i in range(0, text_seq_length - 1):
            words_embeddings_coarse[:,i,:] = (words_embeddings_fine[:,i,:] + words_embeddings_fine[:,i + 1,:]) / 2
        # words_embeddings_coarse
        # if position_ids is None:
        position_ids_fine = torch.arange(text_seq_length, dtype=torch.long, device=input_ids.device)
        position_ids_fine = position_ids_fine.unsqueeze(0).expand_as(input_ids)
        position_embeddings_fine = self.position_embeddings(position_ids_fine)
        position_embeddings_coarse = position_embeddings_fine[:,:-1,:]
        # token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings_fine = words_embeddings_fine + position_embeddings_fine
        embeddings_coarse = words_embeddings_coarse + position_embeddings_coarse
        if self.fp32_embedding:
            embeddings_fine = embeddings_fine.half()
            embeddings_coarse = embeddings_coarse.half()
        embeddings_fine = self.LayerNorm(embeddings_fine)
        embeddings_fine = self.dropout(embeddings_fine)
        embeddings_coarse = self.LayerNorm(embeddings_coarse)
        embeddings_coarse = self.dropout(embeddings_coarse)
        return embeddings_fine,embeddings_coarse


class BertPositionTokenTypeEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """

    def __init__(self, config):
        super(BertPositionTokenTypeEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(
            config.type_vocab_size, config.hidden_size)
        if hasattr(config, 'fp32_embedding'):
            self.fp32_embedding = config.fp32_embedding
        else:
            self.fp32_embedding = False

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-5)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, vis_feats, text_feats,position_ids):

        token_type_ids = torch.zeros(text_feats.size(0),text_feats.size(1) + vis_feats.size(1), dtype=torch.long ,device=text_feats.device)
        # 第一个[SEP]及之前为text，设为0，后面设为1
        # token_type_ids[:,vis_feats.size(1):].fill_(1)
        token_type_ids[:,text_feats.size(1):].fill_(1)
        # position_ids = torch.cat((text_pe[:, :1], vis_pe,
        #                                      text_pe[:, 1:]), dim=1) 
        position_embeddings = self.position_embeddings(position_ids)
       
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        words_embeddings = torch.cat([
            text_feats,
            vis_feats,
            ],dim=1)
        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        if self.fp32_embedding:
            embeddings = embeddings.half()
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

# 没有文本信息的情况
#     def forward(self, vis_feats, vis_pe, input_ids, token_type_ids=None, position_ids=None, vis_input=True,
#                 len_vis_input=49):
#         seq_length = input_ids.size(1)
#         # if position_ids is None:
#         #     position_ids = torch.arange(
#         #         seq_length, dtype=torch.long, device=input_ids.device)
#         #     position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
#         # if token_type_ids is None:
#         #     token_type_ids = torch.zeros_like(input_ids)

#         # words_embeddings = self.word_embeddings(input_ids)
#         words_embeddings = vis_feats
#         position_embeddings = self.position_embeddings(vis_pe)
#         # if vis_input and len_vis_input != 0:
#         #     words_embeddings = torch.cat((words_embeddings[:, :1], vis_feats,
#         #                                   words_embeddings[:, len_vis_input + 1:]), dim=1)
#         #     # assert len_vis_input == 100, 'only support region attn!'

#         #     # 新增的，原版中vis_pe在传入此方法前已经算好
#         #     vis_pe_embedding = self.position_embeddings(vis_pe)
#         #     position_embeddings = torch.cat((position_embeddings[:, :1], vis_pe_embedding,
#         #                                      position_embeddings[:, len_vis_input + 1:]), dim=1)  # hacky...
#         # token_type_embeddings = self.token_type_embeddings(token_type_ids)

# # 去除token embedding
#         embeddings = words_embeddings + position_embeddings 
#         if self.fp32_embedding:
#             embeddings = embeddings.half()
#         embeddings = self.LayerNorm(embeddings)
#         embeddings = self.dropout(embeddings)
#         return embeddings

class BertRespectiveEmbeddings(nn.Module):
    """用于Hybrid Transformer，分别对视频和文本加上position embedding
    """

    def __init__(self, config):
        super(BertRespectiveEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(
            config.type_vocab_size, config.hidden_size)
        if hasattr(config, 'fp32_embedding'):
            self.fp32_embedding = config.fp32_embedding
        else:
            self.fp32_embedding = False

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-5)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, vis_feats, text_feats,position_ids):

        # token_type_ids = torch.zeros(text_feats.size(0),text_feats.size(1) + vis_feats.size(1), dtype=torch.long ,device=text_feats.device)
        # # 第一个[SEP]及之前为text，设为0，后面设为1
        # # token_type_ids[:,vis_feats.size(1):].fill_(1)
        # token_type_ids[:,text_feats.size(1):].fill_(1)
        position_embeddings_vis = self.position_embeddings(position_ids)
        batch = text_feats.size(0)
        text_seq_length = text_feats.size(1)
        position_ids_text = torch.arange(text_seq_length, dtype=torch.long, device=text_feats.device)
        position_ids_text = position_ids_text.expand(batch, text_seq_length)
        position_embeddings_text = self.position_embeddings(position_ids_text)
       
        embeddings_vis = vis_feats + position_embeddings_vis
        embeddings_text = text_feats + position_embeddings_text
        if self.fp32_embedding:
            embeddings_vis = embeddings_vis.half()
            embeddings_text = embeddings_text.half()
        embeddings_vis = self.LayerNorm(embeddings_vis)
        embeddings_vis = self.dropout(embeddings_vis)
        embeddings_text = self.LayerNorm(embeddings_text)
        embeddings_text = self.dropout(embeddings_text)
        return embeddings_vis, embeddings_text



class BertEmbeddingsSignTokenization(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """

    def __init__(self, config):
        super(BertEmbeddingsSignTokenization, self).__init__()
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(
            3, config.hidden_size)
        # 0-frame 1-lefthand 2-righthand 3-head
        self.token_type_embeddings = nn.Embedding(
            4, config.hidden_size)
        if hasattr(config, 'fp32_embedding'):
            self.fp32_embedding = config.fp32_embedding
        else:
            self.fp32_embedding = False

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-5)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, vis_feats,vis_pe,token_type_ids):
       
        words_embeddings = vis_feats
        # vis_pe = [0,0,0,0,1,1,1,1,2,2,2,2]
        vis_pe = torch.tensor(vis_pe,device=vis_feats.device)
        vis_pe = vis_pe.expand(vis_feats.size(0),vis_pe.size(0))
        position_embeddings = self.position_embeddings(vis_pe)
        # token_type_ids = [0,1,2,3,0,1,2,3,0,1,2,3]
        if token_type_ids == None:
            embeddings = words_embeddings + position_embeddings
        else:
            token_type_ids = torch.tensor(token_type_ids,device=vis_feats.device) 
            token_type_ids = token_type_ids.expand(vis_feats.size(0),token_type_ids.size(0))
            token_type_embeddings = self.token_type_embeddings(token_type_ids)
            embeddings = words_embeddings + position_embeddings + token_type_embeddings
        if self.fp32_embedding:
            embeddings = embeddings.half()
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

        


class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super(BertSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(
            config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[
                      :-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask, history_states=None):
        if history_states is None:
            mixed_query_layer = self.query(hidden_states)
            mixed_key_layer = self.key(hidden_states)
            mixed_value_layer = self.value(hidden_states)
        else:
            x_states = torch.cat((history_states, hidden_states), dim=1)
            mixed_query_layer = self.query(hidden_states)
            mixed_key_layer = self.key(x_states)
            mixed_value_layer = self.value(x_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(
            query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / \
                           math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[
                                  :-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer

class BertCrossAttention(nn.Module):
    def __init__(self, config):
        super(BertCrossAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(
            config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        # new_x_shape : (bs,length,num_attention_heads,attention_head_size)
        new_x_shape = x.size()[
                      :-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states_Q, hidden_states_K_V, attention_mask, history_states=None):
        if history_states is None:
            mixed_query_layer = self.query(hidden_states_Q)
            mixed_key_layer = self.key(hidden_states_K_V)
            mixed_value_layer = self.value(hidden_states_K_V)
        else:
            x_states = torch.cat((history_states, hidden_states_K_V), dim=1)
            mixed_query_layer = self.query(hidden_states_Q)
            mixed_key_layer = self.key(x_states)
            mixed_value_layer = self.value(x_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(
            query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / \
                           math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[
                                  :-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer





class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-5)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):
    def __init__(self, config):
        super(BertAttention, self).__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, input_tensor, attention_mask, history_states=None):
        self_output = self.self(
            input_tensor, attention_mask, history_states=history_states)
        attention_output = self.output(self_output, input_tensor)
        return attention_output

class BertHybridAttention(nn.Module):
    def __init__(self, config):
        super(BertHybridAttention, self).__init__()
        self.self = BertSelfAttention(config)
        self.cross = BertCrossAttention(config)
        self.output_1 = BertSelfOutput(config)
        self.output_2 = BertSelfOutput(config)

    def forward(self, query,key_value, attention_mask_query,attention_mask_cross, history_states=None):
        self_output = self.self(
            query, attention_mask_query, history_states=history_states)
        hidden_states_Q = self.output_1(self_output, query)
        cross_output = self.cross( hidden_states_Q, key_value, attention_mask_cross, history_states=None)
        attention_output = self.output_2(cross_output,hidden_states_Q)
        return attention_output


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = ACT2FN[config.hidden_act] \
            if isinstance(config.hidden_act, str) else config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-5)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class BertIntermediateNoConfig(nn.Module):
    def __init__(self, hidden_size,intermediate_size,hidden_act):
        super(BertIntermediateNoConfig, self).__init__()
        self.dense = nn.Linear(hidden_size, intermediate_size)
        self.intermediate_act_fn = ACT2FN[hidden_act] \
            if isinstance(hidden_act, str) else hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutputNoConfig(nn.Module):
    def __init__(self, hidden_size,intermediate_size,hidden_dropout_prob):
        super(BertOutputNoConfig, self).__init__()
        self.dense = nn.Linear(intermediate_size, hidden_size)
        self.LayerNorm = BertLayerNorm(hidden_size, eps=1e-5)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):
    def __init__(self, config):
        super(BertLayer, self).__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask, history_states=None):
        attention_output = self.attention(
            hidden_states, attention_mask, history_states=history_states)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output

class BertHybridLayer(nn.Module):
    def __init__(self, config):
        super(BertHybridLayer, self).__init__()
        self.attention = BertHybridAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states_Q, hidden_states_K_V, attention_mask_query, attention_mask_cross, history_states=None):
        attention_output = self.attention(
            hidden_states_Q, hidden_states_K_V, attention_mask_query, attention_mask_cross, history_states=history_states)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output

class BertHybridEncoder(nn.Module):
    def __init__(self, config):
        super(BertHybridEncoder, self).__init__()
        layer = BertHybridLayer(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer)
                                    for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states_1, hidden_states_2, attention_mask_1, attention_mask_2,attention_mask_cross, prev_embedding=None, prev_encoded_layers=None,
                output_all_encoded_layers=True):
        assert (prev_embedding is None) == (prev_encoded_layers is None), \
            "history embedding and encoded layer must be simultanously given."
        all_encoder_layers = []
        attention_mask_cross_2to1 = attention_mask_cross.transpose(2,3)
        if (prev_embedding is not None) and (prev_encoded_layers is not None):
            history_states = prev_embedding
            for i, layer_module in enumerate(self.layer):
                hidden_states_1 = layer_module(
                    hidden_states_1, hidden_states_2, attention_mask_1, attention_mask_cross, history_states=history_states)
                hidden_states_2 = layer_module(
                    hidden_states_2, hidden_states_1, attention_mask_2, attention_mask_cross_2to1, history_states=history_states)
                if output_all_encoded_layers:
                    all_encoder_layers.append([hidden_states_1, hidden_states_2])
                if prev_encoded_layers is not None:
                    history_states = prev_encoded_layers[i]
        else:
            for layer_module in self.layer:
                hidden_states_1 = layer_module(
                    hidden_states_1, hidden_states_2, attention_mask_1, attention_mask_cross)
                hidden_states_2 = layer_module(
                    hidden_states_2, hidden_states_1, attention_mask_2, attention_mask_cross_2to1)
                if output_all_encoded_layers:
                    all_encoder_layers.append([hidden_states_1, hidden_states_2])
        if not output_all_encoded_layers:
            all_encoder_layers.append([hidden_states_1, hidden_states_2])
        return all_encoder_layers


class BertHybridEncoderNoShare(nn.Module):
    def __init__(self, config):
        super(BertHybridEncoderNoShare, self).__init__()
        layer = BertHybridLayer(config)
        self.layer_1 = nn.ModuleList([copy.deepcopy(layer)
                                    for _ in range(config.num_hidden_layers)])
        self.layer_2 = nn.ModuleList([copy.deepcopy(layer)
                                    for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states_1, hidden_states_2, attention_mask_1, attention_mask_2,attention_mask_cross, prev_embedding=None, prev_encoded_layers=None,
                output_all_encoded_layers=True):
        assert (prev_embedding is None) == (prev_encoded_layers is None), \
            "history embedding and encoded layer must be simultanously given."
        all_encoder_layers = []
        attention_mask_cross_2to1 = attention_mask_cross.transpose(2,3)
        if (prev_embedding is not None) and (prev_encoded_layers is not None):
            history_states = prev_embedding
            for i, layer_module in enumerate(self.layer_1):
                hidden_states_1 = layer_module(
                    hidden_states_1, hidden_states_2, attention_mask_1, attention_mask_cross, history_states=history_states)
                hidden_states_2 = self.layer_2[i](
                    hidden_states_2, hidden_states_1, attention_mask_2, attention_mask_cross_2to1, history_states=history_states)
                if output_all_encoded_layers:
                    all_encoder_layers.append([hidden_states_1, hidden_states_2])
                if prev_encoded_layers is not None:
                    history_states = prev_encoded_layers[i]
        else:
            for layer_module_1,layer_module_2 in zip(self.layer_1,self.layer_2):
                hidden_states_1 = layer_module_1(
                    hidden_states_1, hidden_states_2, attention_mask_1, attention_mask_cross)
                hidden_states_2 = layer_module_2(
                    hidden_states_2, hidden_states_1, attention_mask_2, attention_mask_cross_2to1)
                if output_all_encoded_layers:
                    all_encoder_layers.append([hidden_states_1, hidden_states_2])
        if not output_all_encoded_layers:
            all_encoder_layers.append([hidden_states_1, hidden_states_2])
        return all_encoder_layers


class BertEncoder(nn.Module):
    def __init__(self, config):
        super(BertEncoder, self).__init__()
        layer = BertLayer(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer)
                                    for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask, prev_embedding=None, prev_encoded_layers=None,
                output_all_encoded_layers=True):
        assert (prev_embedding is None) == (prev_encoded_layers is None), \
            "history embedding and encoded layer must be simultanously given."
        all_encoder_layers = []
        if (prev_embedding is not None) and (prev_encoded_layers is not None):
            history_states = prev_embedding
            for i, layer_module in enumerate(self.layer):
                hidden_states = layer_module(
                    hidden_states, attention_mask, history_states=history_states)
                if output_all_encoded_layers:
                    all_encoder_layers.append(hidden_states)
                if prev_encoded_layers is not None:
                    history_states = prev_encoded_layers[i]
        else:
            for layer_module in self.layer:
                hidden_states = layer_module(hidden_states, attention_mask)
                if output_all_encoded_layers:
                    all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers




class BertPooler(nn.Module):
    def __init__(self, config):
        super(BertPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super(BertPredictionHeadTransform, self).__init__()
        self.transform_act_fn = ACT2FN[config.hidden_act] \
            if isinstance(config.hidden_act, str) else config.hidden_act
        hid_size = config.hidden_size
        if hasattr(config, 'relax_projection') and (config.relax_projection > 1):
            hid_size *= config.relax_projection
        self.dense = nn.Linear(config.hidden_size, hid_size)
        self.LayerNorm = BertLayerNorm(hid_size, eps=1e-5)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertLMPredictionHead(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        super(BertLMPredictionHead, self).__init__()
        self.transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(bert_model_embedding_weights.size(1),
                                 bert_model_embedding_weights.size(0),
                                 bias=False)
        self.decoder.weight = bert_model_embedding_weights
        self.bias = nn.Parameter(torch.zeros(
            bert_model_embedding_weights.size(0)))
        if hasattr(config, 'relax_projection') and (config.relax_projection > 1):
            self.relax_projection = config.relax_projection
        else:
            self.relax_projection = 0
        self.fp32_embedding = config.fp32_embedding

        def convert_to_type(tensor):
            if self.fp32_embedding:
                return tensor.half()
            else:
                return tensor

        self.type_converter = convert_to_type
        self.converted = False

    def forward(self, hidden_states, task_idx=None):
        if not self.converted:
            self.converted = True
            if self.fp32_embedding:
                self.transform.half()
        hidden_states = self.transform(self.type_converter(hidden_states))
        if self.relax_projection > 1:
            num_batch = hidden_states.size(0)
            num_pos = hidden_states.size(1)
            # (batch, num_pos, relax_projection*hid) -> (batch, num_pos, relax_projection, hid) -> (batch, num_pos, hid)
            hidden_states = hidden_states.view(
                num_batch, num_pos, self.relax_projection, -1)[torch.arange(0, num_batch).long(), :, task_idx, :]
        if self.fp32_embedding:
            hidden_states = F.linear(self.type_converter(hidden_states), self.type_converter(
                self.decoder.weight), self.type_converter(self.bias))
        else:
            hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states


class BertOnlyMLMHead(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        super(BertOnlyMLMHead, self).__init__()
        self.predictions = BertLMPredictionHead(
            config, bert_model_embedding_weights)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class BertOnlyNSPHead(nn.Module):
    def __init__(self, config):
        super(BertOnlyNSPHead, self).__init__()
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, pooled_output):
        seq_relationship_score = self.seq_relationship(pooled_output)
        return seq_relationship_score


class BertPreTrainingHeads(nn.Module):
    def __init__(self, config, bert_model_embedding_weights, num_labels=2):
        super(BertPreTrainingHeads, self).__init__()
        self.predictions = BertLMPredictionHead(
            config, bert_model_embedding_weights)
        self.seq_relationship = nn.Linear(config.hidden_size, num_labels)

    def forward(self, sequence_output, pooled_output, task_idx=None):
        prediction_scores = self.predictions(sequence_output, task_idx)
        if pooled_output is None:
            seq_relationship_score = None
        else:
            seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score


class PreTrainedBertModel(nn.Module):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """

    def __init__(self, config, *inputs, **kwargs):
        super(PreTrainedBertModel, self).__init__()
        if not isinstance(config, BertConfig):
            raise ValueError(
                "Parameter config in `{}(config)` should be an instance of class `BertConfig`. "
                "To create a model from a Google pretrained model use "
                "`model = {}.from_pretrained(PRETRAINED_MODEL_NAME)`".format(
                    self.__class__.__name__, self.__class__.__name__
                ))
        self.config = config

    def init_bert_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(
                mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    @classmethod
    def from_pretrained(cls, pretrained_model_name, state_dict=None, cache_dir=None, *inputs, **kwargs):
        """
        Instantiate a PreTrainedBertModel from a pre-trained model file or a pytorch state dict.
        Download and cache the pre-trained model file if needed.

        Params:
            pretrained_model_name: either:
                - a str with the name of a pre-trained model to load selected in the list of:
                    . `bert-base-uncased`
                    . `bert-large-uncased`
                    . `bert-base-cased`
                    . `bert-base-multilingual`
                    . `bert-base-chinese`
                - a path or url to a pretrained model archive containing:
                    . `bert_config.json` a configuration file for the model
                    . `pytorch_model.bin` a PyTorch dump of a BertForPreTraining instance
            cache_dir: an optional path to a folder in which the pre-trained models will be cached.
            state_dict: an optional state dictionnary (collections.OrderedDict object) to use instead of Google pre-trained models
            *inputs, **kwargs: additional input for the specific Bert class
                (ex: num_labels for BertForSequenceClassification)
        """
        print('init start')
        if pretrained_model_name in PRETRAINED_MODEL_ARCHIVE_MAP:
            archive_file = PRETRAINED_MODEL_ARCHIVE_MAP[pretrained_model_name]
        else:
            archive_file = pretrained_model_name
        # redirect to the cache, if necessary
        try:
            resolved_archive_file = cached_path(
                archive_file, cache_dir=cache_dir)
        except FileNotFoundError:
            logger.error(
                "Model name '{}' was not found in model name list ({}). "
                "We assumed '{}' was a path or url but couldn't find any file "
                "associated to this path or url.".format(
                    pretrained_model_name,
                    ', '.join(PRETRAINED_MODEL_ARCHIVE_MAP.keys()),
                    archive_file))
            return None
        if resolved_archive_file == archive_file:
            logger.info("loading archive file {}".format(archive_file))
        else:
            logger.info("loading archive file {} from cache at {}".format(
                archive_file, resolved_archive_file))
        tempdir = None
        if os.path.isdir(resolved_archive_file):
            serialization_dir = resolved_archive_file
        else:
            # Extract archive to temp dir
            tempdir = tempfile.mkdtemp()
            logger.info("extracting archive file {} to temp dir {}".format(
                resolved_archive_file, tempdir))
            with tarfile.open(resolved_archive_file, 'r:gz') as archive:
                archive.extractall(tempdir)
            serialization_dir = tempdir
        # Load config
        if ('config_path' in kwargs) and kwargs['config_path']:
            config_file = kwargs['config_path']
        else:
            config_file = os.path.join(serialization_dir, CONFIG_NAME)
        config = BertConfig.from_json_file(config_file)

        # define new type_vocab_size (there might be different numbers of segment ids)
        if 'type_vocab_size' in kwargs:
            config.type_vocab_size = kwargs['type_vocab_size']
        if 'vocab_size' in kwargs:
            config.vocab_size = kwargs['vocab_size']
        # define new relax_projection
        if ('relax_projection' in kwargs) and kwargs['relax_projection']:
            config.relax_projection = kwargs['relax_projection']
        # define new relax_projection
        if ('task_idx' in kwargs) and kwargs['task_idx']:
            config.task_idx = kwargs['task_idx']
        # define new max position embedding for length expansion
        if ('max_position_embeddings' in kwargs) and kwargs['max_position_embeddings']:
            config.max_position_embeddings = kwargs['max_position_embeddings']
        # use fp32 for embeddings
        if ('fp32_embedding' in kwargs) and kwargs['fp32_embedding']:
            config.fp32_embedding = kwargs['fp32_embedding']
        # label smoothing
        if ('label_smoothing' in kwargs) and kwargs['label_smoothing']:
            config.label_smoothing = kwargs['label_smoothing']
        if 'drop_prob' in kwargs:
            print('setting the new dropout rate!', kwargs['drop_prob'])
            config.attention_probs_dropout_prob = kwargs['drop_prob']
            config.hidden_dropout_prob = kwargs['drop_prob']

        if 'num_hidden_layers' in kwargs:
            print('setting the new num_hidden_layers!', kwargs['num_hidden_layers'])
            config.num_hidden_layers = kwargs['num_hidden_layers']

        logger.info("Model config {}".format(config))

        # 只用2层
        # config.num_hidden_layers = 4

        # clean the arguments in kwargs
        for arg_clean in (
                'config_path', 'type_vocab_size', 'relax_projection', 'task_idx', 'max_position_embeddings',
                'fp32_embedding',
                'label_smoothing', 'drop_prob' ,'num_hidden_layers'):
            if arg_clean in kwargs:
                del kwargs[arg_clean]

        # Instantiate model.
        model = cls(config, *inputs, **kwargs)
        if state_dict is None:
            weights_path = os.path.join(serialization_dir, WEIGHTS_NAME)
            state_dict = torch.load(weights_path)

        old_keys = []
        new_keys = []
        for key in state_dict.keys():
            new_key = None
            if 'gamma' in key:
                new_key = key.replace('gamma', 'weight')
            if 'beta' in key:
                new_key = key.replace('beta', 'bias')
            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)
        for old_key, new_key in zip(old_keys, new_keys):
            state_dict[new_key] = state_dict.pop(old_key)

        # initialize new segment embeddings
        _k = 'bert.embeddings.token_type_embeddings.weight'
        if (_k in state_dict) and (config.type_vocab_size != state_dict[_k].shape[0]):
            logger.info(
                "config.type_vocab_size != state_dict[bert.embeddings.token_type_embeddings.weight] ({0} != {1})".format(
                    config.type_vocab_size, state_dict[_k].shape[0]))
            if config.type_vocab_size > state_dict[_k].shape[0]:
                # state_dict[_k].data = state_dict[_k].data.resize_(
                state_dict[_k].data = state_dict[_k].resize_(
                    config.type_vocab_size, state_dict[_k].shape[1]).data
                if config.type_vocab_size >= 6:
                    # L2R
                    state_dict[_k].data[2, :].copy_(state_dict[_k].data[0, :])
                    # R2L
                    state_dict[_k].data[3, :].copy_(state_dict[_k].data[0, :])
                    # S2S
                    state_dict[_k].data[4, :].copy_(state_dict[_k].data[0, :])
                    state_dict[_k].data[5, :].copy_(state_dict[_k].data[1, :])
            elif config.type_vocab_size < state_dict[_k].shape[0]:
                state_dict[_k].data = state_dict[_k].data[:config.type_vocab_size, :]

        # initialize new position embeddings
        _k = 'bert.embeddings.position_embeddings.weight'
        if _k in state_dict and config.max_position_embeddings != state_dict[_k].shape[0]:
            logger.info(
                "config.max_position_embeddings != state_dict[bert.embeddings.position_embeddings.weight] ({0} - {1})".format(
                    config.max_position_embeddings, state_dict[_k].shape[0]))
            if config.max_position_embeddings > state_dict[_k].shape[0]:
                old_size = state_dict[_k].shape[0]
                state_dict[_k].data = state_dict[_k].data.resize_(
                    config.max_position_embeddings, state_dict[_k].shape[1])
                start = old_size
                while start < config.max_position_embeddings:
                    chunk_size = min(
                        old_size, config.max_position_embeddings - start)
                    state_dict[_k].data[start:start + chunk_size,
                    :].copy_(state_dict[_k].data[:chunk_size, :])
                    start += chunk_size
            elif config.max_position_embeddings < state_dict[_k].shape[0]:
                state_dict[_k].data = state_dict[_k].data[:config.max_position_embeddings, :]

        # initialize relax projection
        _k = 'cls.predictions.transform.dense.weight'
        n_config_relax = 1 if (config.relax_projection <
                               1) else config.relax_projection
        if (_k in state_dict) and (n_config_relax * config.hidden_size != state_dict[_k].shape[0]):
            logger.info(
                "n_config_relax*config.hidden_size != state_dict[cls.predictions.transform.dense.weight] ({0}*{1} != {2})".format(
                    n_config_relax, config.hidden_size, state_dict[_k].shape[0]))
            assert state_dict[_k].shape[0] % config.hidden_size == 0
            n_state_relax = state_dict[_k].shape[0] // config.hidden_size
            assert (n_state_relax == 1) != (n_config_relax ==
                                            1), "!!!!n_state_relax == 1 xor n_config_relax == 1!!!!"
            if n_state_relax == 1:
                _k = 'cls.predictions.transform.dense.weight'
                state_dict[_k].data = state_dict[_k].data.unsqueeze(0).repeat(
                    n_config_relax, 1, 1).reshape((n_config_relax * config.hidden_size, config.hidden_size))
                for _k in ('cls.predictions.transform.dense.bias', 'cls.predictions.transform.LayerNorm.weight',
                           'cls.predictions.transform.LayerNorm.bias'):
                    state_dict[_k].data = state_dict[_k].data.unsqueeze(
                        0).repeat(n_config_relax, 1).view(-1)
            elif n_config_relax == 1:
                if hasattr(config, 'task_idx') and (config.task_idx is not None) and (0 <= config.task_idx <= 3):
                    _task_idx = config.task_idx
                else:
                    _task_idx = 0
                _k = 'cls.predictions.transform.dense.weight'
                state_dict[_k].data = state_dict[_k].data.view(
                    n_state_relax, config.hidden_size, config.hidden_size).select(0, _task_idx)
                for _k in ('cls.predictions.transform.dense.bias', 'cls.predictions.transform.LayerNorm.weight',
                           'cls.predictions.transform.LayerNorm.bias'):
                    state_dict[_k].data = state_dict[_k].data.view(
                        n_state_relax, config.hidden_size).select(0, _task_idx)

        missing_keys = []
        unexpected_keys = []
        error_msgs = []
        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, '_metadata', None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        # use_high_layer 表示使用bert的高层参数初始化低层
        def load(module, prefix='',use_high_layer = None):
            local_metadata = {} if metadata is None else metadata.get(
                prefix[:-1], {})
            module._load_from_state_dict(
                state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
            for name, child in module._modules.items():
                if child is not None:            
                    if use_high_layer is not None:
                        if prefix == 'bert.encoder.layer.0.':
                            prefix = 'bert.encoder.layer.6.'
                        elif prefix == 'bert.encoder.layer.1.':
                            prefix = 'bert.encoder.layer.7.'
                        elif prefix == 'bert.encoder.layer.2.':
                            prefix = 'bert.encoder.layer.8.'
                        elif prefix == 'bert.encoder.layer.3.':
                            prefix = 'bert.encoder.layer.9.'
                        elif prefix == 'bert.encoder.layer.4.':
                            prefix = 'bert.encoder.layer.10.'
                        elif prefix == 'bert.encoder.layer.5.':
                            prefix = 'bert.encoder.layer.11.'

                    load(child, prefix + name + '.',use_high_layer)

        load(model, prefix='' if hasattr(model, 'bert') else 'bert.',use_high_layer=kwargs['use_high_layer'])
        model.missing_keys = missing_keys
        if len(missing_keys) > 0:
            logger.info("Weights of {} not initialized from pretrained model: {}".format(
                model.__class__.__name__, missing_keys))
        if len(unexpected_keys) > 0:
            logger.info("Weights from pretrained model not used in {}: {}".format(
                model.__class__.__name__, unexpected_keys))
        if len(error_msgs) > 0:
            logger.info('\n'.join(error_msgs))
        if tempdir:
            # Clean up temp dir
            shutil.rmtree(tempdir)
        return model


class BertModel(PreTrainedBertModel):
    """BERT model ("Bidirectional Embedding Representations from a Transformer").

    Params:
        config: a BertConfig class instance with the configuration to build a new model

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `output_all_encoded_layers`: boolean which controls the content of the `encoded_layers` output as described below. Default: `True`.

    Outputs: Tuple of (encoded_layers, pooled_output)
        `encoded_layers`: controled by `output_all_encoded_layers` argument:
            - `output_all_encoded_layers=True`: outputs a list of the full sequences of encoded-hidden-states at the end
                of each attention block (i.e. 12 full sequences for BERT-base, 24 for BERT-large), each
                encoded-hidden-state is a torch.FloatTensor of size [batch_size, sequence_length, hidden_size],
            - `output_all_encoded_layers=False`: outputs only the full sequence of hidden-states corresponding
                to the last attention block of shape [batch_size, sequence_length, hidden_size],
        `pooled_output`: a torch.FloatTensor of size [batch_size, hidden_size] which is the output of a
            classifier pretrained on top of the hidden state associated to the first character of the
            input (`CLF`) to train on the Next-Sentence task (see BERT's paper).

    """
#    config 默认 bert-base-cased
    def __init__(self, config):
        super(BertModel, self).__init__(config)
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)
        self.apply(self.init_bert_weights)

    def get_extended_attention_mask(self, input_ids, token_type_ids, attention_mask):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        # if token_type_ids is None:
            # token_type_ids = torch.zeros_like(input_ids)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        if attention_mask.dim() == 2:
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        elif attention_mask.dim() == 3:
            extended_attention_mask = attention_mask.unsqueeze(1)
        else:
            raise NotImplementedError

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def forward(self, vis_feats, vis_pe, input_ids, token_type_ids=None, attention_mask=None,
                output_all_encoded_layers=True, len_vis_input=49):
        extended_attention_mask = self.get_extended_attention_mask(
            input_ids, token_type_ids, attention_mask)

        # hack to load vis feats
        embedding_output = self.embeddings(vis_feats, vis_pe, input_ids, token_type_ids, len_vis_input=len_vis_input)
        encoded_layers = self.encoder(embedding_output,
                                      extended_attention_mask,
                                      output_all_encoded_layers=output_all_encoded_layers)
        sequence_output = encoded_layers[-1]
        pooled_output = self.pooler(sequence_output)
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]
        return encoded_layers, pooled_output



class BertMultiScaleEmbeddingModel(PreTrainedBertModel):
    """BERT model ("Bidirectional Embedding Representations from a Transformer").

    Params:
        config: a BertConfig class instance with the configuration to build a new model

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `output_all_encoded_layers`: boolean which controls the content of the `encoded_layers` output as described below. Default: `True`.

    Outputs: Tuple of (encoded_layers, pooled_output)
        `encoded_layers`: controled by `output_all_encoded_layers` argument:
            - `output_all_encoded_layers=True`: outputs a list of the full sequences of encoded-hidden-states at the end
                of each attention block (i.e. 12 full sequences for BERT-base, 24 for BERT-large), each
                encoded-hidden-state is a torch.FloatTensor of size [batch_size, sequence_length, hidden_size],
            - `output_all_encoded_layers=False`: outputs only the full sequence of hidden-states corresponding
                to the last attention block of shape [batch_size, sequence_length, hidden_size],
        `pooled_output`: a torch.FloatTensor of size [batch_size, hidden_size] which is the output of a
            classifier pretrained on top of the hidden state associated to the first character of the
            input (`CLF`) to train on the Next-Sentence task (see BERT's paper).

    """
#    config 默认 bert-base-cased
    def __init__(self, config):
        super(BertMultiScaleEmbeddingModel, self).__init__(config)
        self.embeddings = BertMultiScaleEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)
        self.apply(self.init_bert_weights)

    def get_extended_attention_mask(self, input_ids, token_type_ids, attention_mask):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        # if token_type_ids is None:
            # token_type_ids = torch.zeros_like(input_ids)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        if attention_mask.dim() == 2:
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        elif attention_mask.dim() == 3:
            extended_attention_mask = attention_mask.unsqueeze(1)
        else:
            raise NotImplementedError

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def forward(self, vis_feats, vis_pe, input_ids, token_type_ids=None, attention_mask=None,attention_mask_coarse=None,
                output_all_encoded_layers=True, len_vis_input=49):
        extended_attention_mask_fine = self.get_extended_attention_mask(
            input_ids, token_type_ids, attention_mask)
        # attention_mask_coarse = attention_mask[:,:-1,:-1]
        extended_attention_mask_coarse = self.get_extended_attention_mask(
            input_ids, token_type_ids, attention_mask_coarse)
        # hack to load vis feats
        embedding_output_fine,embedding_output_coarse = self.embeddings(input_ids)
        encoded_layers_fine = self.encoder(embedding_output_fine,
                                      extended_attention_mask_fine,
                                      output_all_encoded_layers=output_all_encoded_layers)
        encoded_layers_coarse = self.encoder(embedding_output_coarse,
                                      extended_attention_mask_coarse,
                                      output_all_encoded_layers=output_all_encoded_layers)
        # sequence_output_fine = encoded_layers_fine[-1]
        # sequence_output_coarse = encoded_layers_coarse[-1]
        # pooled_output = self.pooler(sequence_output)
        # if not output_all_encoded_layers:
        #     encoded_layers = encoded_layers[-1]
        return encoded_layers_fine[-1], encoded_layers_coarse[-1]


class BertModelSignTokenization(PreTrainedBertModel):
#    config 默认 bert-base-cased
    def __init__(self, config):
        super(BertModelSignTokenization, self).__init__(config)
        self.embeddings = BertEmbeddingsSignTokenization(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)
        self.fc = nn.Linear(3,1)
        self.activation = nn.Tanh()
        self.temporal_pooling = nn.AdaptiveAvgPool1d(1)
        self.apply(self.init_bert_weights)

    def get_extended_attention_mask(self, attention_mask):

        if attention_mask.dim() == 2:
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        elif attention_mask.dim() == 3:
            extended_attention_mask = attention_mask.unsqueeze(1)
        else:
            raise NotImplementedError

        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def forward(self, vis_feats,vis_pe,token_type_ids, attention_mask=None,
                output_all_encoded_layers=True):
        extended_attention_mask = self.get_extended_attention_mask(
            attention_mask)

        # hack to load vis feats
        embedding_output = self.embeddings(vis_feats,vis_pe,token_type_ids)
        encoded_layers = self.encoder(embedding_output,
                                      extended_attention_mask,
                                      output_all_encoded_layers=output_all_encoded_layers)
        sequence_output = encoded_layers[-1]
        # pooled_output = self.pooler(sequence_output)
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]
            encoded_layers = self.temporal_pooling(encoded_layers.transpose(2,1))
            # encoded_layers = self.fc(encoded_layers.transpose(2,1))
            # encoded_layers = self.activation(encoded_layers)
        return encoded_layers.transpose(1,2), []


class BertModelIncr(BertModel):
    def __init__(self, config):
        super(BertModelIncr, self).__init__(config)

    def forward(self, vis_feats, vis_pe, input_ids, token_type_ids, position_ids, attention_mask,
                prev_embedding=None, prev_encoded_layers=None, output_all_encoded_layers=True,
                len_vis_input=49):
        extended_attention_mask = self.get_extended_attention_mask(
            input_ids, token_type_ids, attention_mask)

        embedding_output = self.embeddings(
            vis_feats, vis_pe, input_ids, token_type_ids, position_ids,
            vis_input=(prev_encoded_layers is None), len_vis_input=len_vis_input)
        encoded_layers = self.encoder(embedding_output,
                                      extended_attention_mask,
                                      prev_embedding=prev_embedding,
                                      prev_encoded_layers=prev_encoded_layers,
                                      output_all_encoded_layers=output_all_encoded_layers)
        sequence_output = encoded_layers[-1]
        pooled_output = self.pooler(sequence_output)
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]
        return embedding_output, encoded_layers, pooled_output


class BertModelAfterEmbedding(BertModel):
    def __init__(self, config):
        super(BertModelAfterEmbedding, self).__init__(config)
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)
        self.apply(self.init_bert_weights)

    def get_extended_attention_mask(self, attention_mask):

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        if attention_mask.dim() == 2:
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        elif attention_mask.dim() == 3:
            extended_attention_mask = attention_mask.unsqueeze(1)
        else:
            raise NotImplementedError
        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def forward(self,embedding_output, attention_mask=None,
                output_all_encoded_layers=True):
        extended_attention_mask = self.get_extended_attention_mask(attention_mask)

        # hack to load vis feats
        # embedding_output = self.embeddings(vis_feats, vis_pe, input_ids, token_type_ids, len_vis_input=len_vis_input)
        encoded_layers = self.encoder(embedding_output,
                                      extended_attention_mask,
                                      output_all_encoded_layers=output_all_encoded_layers)
        sequence_output = encoded_layers[-1]
        pooled_output = self.pooler(sequence_output)
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]
        return encoded_layers, pooled_output


class BertModelFusion(BertModel):
    def __init__(self, config):
        super(BertModelFusion, self).__init__(config)
        self.embeddings = BertPositionTokenTypeEmbeddings(config) # word embedding已经完成
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)
        self.apply(self.init_bert_weights)

    def get_extended_attention_mask(self, attention_mask):

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        if attention_mask.dim() == 2:
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        elif attention_mask.dim() == 3:
            extended_attention_mask = attention_mask.unsqueeze(1)
        else:
            raise NotImplementedError
        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def forward(self,vis_feat,text_feat,position_ids, attention_mask=None,
                output_all_encoded_layers=True):
        extended_attention_mask = self.get_extended_attention_mask(attention_mask)

        # hack to load vis feats
        embedding_output = self.embeddings(vis_feat,text_feat,position_ids)
        encoded_layers = self.encoder(embedding_output,
                                      extended_attention_mask,
                                      output_all_encoded_layers=output_all_encoded_layers)
        sequence_output = encoded_layers[-1]
        pooled_output = self.pooler(sequence_output)
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]
        return encoded_layers, pooled_output

class BertModelHybrid(BertModel):
    def __init__(self, config):
        super(BertModel, self).__init__(config)
        self.embeddings = BertRespectiveEmbeddings(config) # word embedding已经完成
        self.encoder = BertHybridEncoder(config)
        self.pooler = BertPooler(config)
        self.apply(self.init_bert_weights)

    def get_extended_attention_mask(self, attention_mask):
        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        if attention_mask.dim() == 2:
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        elif attention_mask.dim() == 3:
            extended_attention_mask = attention_mask.unsqueeze(1)
        else:
            raise NotImplementedError
        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def forward(self,vis_feat,text_feat,position_ids, attention_mask_vis, attention_mask_text, attention_mask_cross_VasQ,
                output_all_encoded_layers=True):
        extended_attention_mask_vis = self.get_extended_attention_mask(attention_mask_vis)
        extended_attention_mask_text = self.get_extended_attention_mask(attention_mask_text)
        extended_attention_mask_cross_VasQ = self.get_extended_attention_mask(attention_mask_cross_VasQ)
        # hack to load vis feats
        embedding_output_vis, embedding_output_text = self.embeddings(vis_feat,text_feat,position_ids)
        encoded_layers = self.encoder(embedding_output_vis,
                                      embedding_output_text,
                                      extended_attention_mask_vis,
                                      extended_attention_mask_text,
                                      extended_attention_mask_cross_VasQ,
                                      output_all_encoded_layers=output_all_encoded_layers)
        sequence_output = encoded_layers[-1]
        # pooled_output = self.pooler(sequence_output)
        pooled_output = self.pooler(sequence_output[1])
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]
        return encoded_layers, pooled_output


class BertModelHybridNoShare(BertModel):
    def __init__(self, config):
        super(BertModel, self).__init__(config)
        self.embeddings = BertRespectiveEmbeddings(config) # word embedding已经完成
        self.encoder = BertHybridEncoderNoShare(config)
        self.pooler = BertPooler(config)
        self.apply(self.init_bert_weights)

    def get_extended_attention_mask(self, attention_mask):
        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        if attention_mask.dim() == 2:
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        elif attention_mask.dim() == 3:
            extended_attention_mask = attention_mask.unsqueeze(1)
        else:
            raise NotImplementedError
        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def forward(self,vis_feat,text_feat,position_ids, attention_mask_vis, attention_mask_text, attention_mask_cross_VasQ,
                output_all_encoded_layers=True):
        extended_attention_mask_vis = self.get_extended_attention_mask(attention_mask_vis)
        extended_attention_mask_text = self.get_extended_attention_mask(attention_mask_text)
        extended_attention_mask_cross_VasQ = self.get_extended_attention_mask(attention_mask_cross_VasQ)
        # hack to load vis feats
        embedding_output_vis, embedding_output_text = self.embeddings(vis_feat,text_feat,position_ids)
        encoded_layers = self.encoder(embedding_output_vis,
                                      embedding_output_text,
                                      extended_attention_mask_vis,
                                      extended_attention_mask_text,
                                      extended_attention_mask_cross_VasQ,
                                      output_all_encoded_layers=output_all_encoded_layers)
        sequence_output = encoded_layers[-1]
        # pooled_output = self.pooler(sequence_output)
        pooled_output = self.pooler(sequence_output[1])
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]
        return encoded_layers, pooled_output

class BertForPreTraining(PreTrainedBertModel):
    """BERT model with pre-training heads.
    This module comprises the BERT model followed by the two pre-training heads:
        - the masked language modeling head, and
        - the next sentence classification head.
    Params:
        config: a BertConfig class instance with the configuration to build a new model.
    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `masked_lm_labels`: masked language modeling labels: torch.LongTensor of shape [batch_size, sequence_length]
            with indices selected in [-1, 0, ..., vocab_size]. All labels set to -1 are ignored (masked), the loss
            is only computed for the labels set in [0, ..., vocab_size]
        `next_sentence_label`: next sentence classification loss: torch.LongTensor of shape [batch_size]
            with indices selected in [0, 1].
            0 => next sentence is the continuation, 1 => next sentence is a random sentence.
    Outputs:
        if `masked_lm_labels` and `next_sentence_label` are not `None`:
            Outputs the total_loss which is the sum of the masked language modeling loss and the next
            sentence classification loss.
        if `masked_lm_labels` or `next_sentence_label` is `None`:
            Outputs a tuple comprising
            - the masked language modeling logits of shape [batch_size, sequence_length, vocab_size], and
            - the next sentence classification logits of shape [batch_size, 2].
    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])
    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)
    model = BertForPreTraining(config)
    masked_lm_logits_scores, seq_relationship_logits = model(input_ids, token_type_ids, input_mask)
    ```
    """

    def __init__(self, config):
        super(BertForPreTraining, self).__init__(config)
        self.bert = BertModel(config)
        self.cls = BertPreTrainingHeads(
            config, self.bert.embeddings.word_embeddings.weight)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, masked_lm_labels=None,
                next_sentence_label=None):
        sequence_output, pooled_output = self.bert(input_ids, token_type_ids, attention_mask,
                                                   output_all_encoded_layers=False)
        prediction_scores, seq_relationship_score = self.cls(
            sequence_output, pooled_output)

        if masked_lm_labels is not None and next_sentence_label is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            masked_lm_loss = loss_fct(
                prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
            next_sentence_loss = loss_fct(
                seq_relationship_score.view(-1, 2), next_sentence_label.view(-1))
            total_loss = masked_lm_loss + next_sentence_loss
            return total_loss
        else:
            return prediction_scores, seq_relationship_score


class BertPreTrainingPairTransform(nn.Module):
    def __init__(self, config):
        super(BertPreTrainingPairTransform, self).__init__()
        self.dense = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.transform_act_fn = ACT2FN[config.hidden_act] \
            if isinstance(config.hidden_act, str) else config.hidden_act
        # self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-5)

    def forward(self, pair_x, pair_y):
        hidden_states = torch.cat([pair_x, pair_y], dim=-1)
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        # hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertPreTrainingPairRel(nn.Module):
    def __init__(self, config, num_rel=0):
        super(BertPreTrainingPairRel, self).__init__()
        self.R_xy = BertPreTrainingPairTransform(config)
        self.rel_emb = nn.Embedding(num_rel, config.hidden_size)

    def forward(self, pair_x, pair_y, pair_r, pair_pos_neg_mask):
        # (batch, num_pair, hidden)
        xy = self.R_xy(pair_x, pair_y)
        r = self.rel_emb(pair_r)
        _batch, _num_pair, _hidden = xy.size()
        pair_score = (xy * r).sum(-1)
        # torch.bmm(xy.view(-1, 1, _hidden),r.view(-1, _hidden, 1)).view(_batch, _num_pair)
        # .mul_(-1.0): objective to loss
        return F.logsigmoid(pair_score * pair_pos_neg_mask.type_as(pair_score)).mul_(-1.0)



class BertForMaskedLM(PreTrainedBertModel):
    """BERT model with the masked language modeling head.
    This module comprises the BERT model followed by the masked language modeling head.

    Params:
        config: a BertConfig class instance with the configuration to build a new model.

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `masked_lm_labels`: masked language modeling labels: torch.LongTensor of shape [batch_size, sequence_length]
            with indices selected in [-1, 0, ..., vocab_size]. All labels set to -1 are ignored (masked), the loss
            is only computed for the labels set in [0, ..., vocab_size]

    Outputs:
        if `masked_lm_labels` is `None`:
            Outputs the masked language modeling loss.
        if `masked_lm_labels` is `None`:
            Outputs the masked language modeling logits of shape [batch_size, sequence_length, vocab_size].

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    model = BertForMaskedLM(config)
    masked_lm_logits_scores = model(input_ids, token_type_ids, input_mask)
    ```
    """

    def __init__(self, config):
        super(BertForMaskedLM, self).__init__(config)
        self.bert = BertModel(config)
        self.cls = BertOnlyMLMHead(
            config, self.bert.embeddings.word_embeddings.weight)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, masked_lm_labels=None):
        sequence_output, _ = self.bert(input_ids, token_type_ids, attention_mask,
                                       output_all_encoded_layers=False)
        prediction_scores = self.cls(sequence_output)

        if masked_lm_labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            masked_lm_loss = loss_fct(
                prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
            return masked_lm_loss
        else:
            return prediction_scores


class BertForNextSentencePrediction(PreTrainedBertModel):
    """BERT model with next sentence prediction head.
    This module comprises the BERT model followed by the next sentence classification head.

    Params:
        config: a BertConfig class instance with the configuration to build a new model.

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `next_sentence_label`: next sentence classification loss: torch.LongTensor of shape [batch_size]
            with indices selected in [0, 1].
            0 => next sentence is the continuation, 1 => next sentence is a random sentence.

    Outputs:
        if `next_sentence_label` is not `None`:
            Outputs the total_loss which is the sum of the masked language modeling loss and the next
            sentence classification loss.
        if `next_sentence_label` is `None`:
            Outputs the next sentence classification logits of shape [batch_size, 2].

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    model = BertForNextSentencePrediction(config)
    seq_relationship_logits = model(input_ids, token_type_ids, input_mask)
    ```
    """

    def __init__(self, config):
        super(BertForNextSentencePrediction, self).__init__(config)
        self.bert = BertModel(config)
        self.cls = BertOnlyNSPHead(config)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, next_sentence_label=None):
        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask,
                                     output_all_encoded_layers=False)
        seq_relationship_score = self.cls(pooled_output)

        if next_sentence_label is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            next_sentence_loss = loss_fct(
                seq_relationship_score.view(-1, 2), next_sentence_label.view(-1))
            return next_sentence_loss
        else:
            return seq_relationship_score


class BertForSequenceClassification(PreTrainedBertModel):
    """BERT model for classification.
    This module is composed of the BERT model with a linear layer on top of
    the pooled output.

    Params:
        `config`: a BertConfig class instance with the configuration to build a new model.
        `num_labels`: the number of classes for the classifier. Default = 2.

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `labels`: labels for the classification output: torch.LongTensor of shape [batch_size]
            with indices selected in [0, ..., num_labels].

    Outputs:
        if `labels` is not `None`:
            Outputs the CrossEntropy classification loss of the output with the labels.
        if `labels` is `None`:
            Outputs the classification logits of shape [batch_size, num_labels].

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    num_labels = 2

    model = BertForSequenceClassification(config, num_labels)
    logits = model(input_ids, token_type_ids, input_mask)
    ```
    """

    def __init__(self, config, num_labels=2):
        super(BertForSequenceClassification, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        _, pooled_output = self.bert(
            input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        if labels is not None:
            if labels.dtype == torch.long:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(
                    logits.view(-1, self.num_labels), labels.view(-1))
            elif labels.dtype == torch.half or labels.dtype == torch.float:
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                print('unkown labels.dtype')
                loss = None
            return loss
        else:
            return logits


class BertForMultipleChoice(PreTrainedBertModel):
    """BERT model for multiple choice tasks.
    This module is composed of the BERT model with a linear layer on top of
    the pooled output.

    Params:
        `config`: a BertConfig class instance with the configuration to build a new model.
        `num_choices`: the number of classes for the classifier. Default = 2.

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, num_choices, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, num_choices, sequence_length]
            with the token types indices selected in [0, 1]. Type 0 corresponds to a `sentence A`
            and type 1 corresponds to a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, num_choices, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `labels`: labels for the classification output: torch.LongTensor of shape [batch_size]
            with indices selected in [0, ..., num_choices].

    Outputs:
        if `labels` is not `None`:
            Outputs the CrossEntropy classification loss of the output with the labels.
        if `labels` is `None`:
            Outputs the classification logits of shape [batch_size, num_labels].

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[[31, 51, 99], [15, 5, 0]], [[12, 16, 42], [14, 28, 57]]])
    input_mask = torch.LongTensor([[[1, 1, 1], [1, 1, 0]],[[1,1,0], [1, 0, 0]]])
    token_type_ids = torch.LongTensor([[[0, 0, 1], [0, 1, 0]],[[0, 1, 1], [0, 0, 1]]])
    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    num_choices = 2

    model = BertForMultipleChoice(config, num_choices)
    logits = model(input_ids, token_type_ids, input_mask)
    ```
    """

    def __init__(self, config, num_choices=2):
        super(BertForMultipleChoice, self).__init__(config)
        self.num_choices = num_choices
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        flat_input_ids = input_ids.view(-1, input_ids.size(-1))
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1))
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1))
        _, pooled_output = self.bert(
            flat_input_ids, flat_token_type_ids, flat_attention_mask, output_all_encoded_layers=False)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, self.num_choices)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)
            return loss
        else:
            return reshaped_logits


class BertForTokenClassification(PreTrainedBertModel):
    """BERT model for token-level classification.
    This module is composed of the BERT model with a linear layer on top of
    the full hidden state of the last layer.

    Params:
        `config`: a BertConfig class instance with the configuration to build a new model.
        `num_labels`: the number of classes for the classifier. Default = 2.

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `labels`: labels for the classification output: torch.LongTensor of shape [batch_size]
            with indices selected in [0, ..., num_labels].

    Outputs:
        if `labels` is not `None`:
            Outputs the CrossEntropy classification loss of the output with the labels.
        if `labels` is `None`:
            Outputs the classification logits of shape [batch_size, sequence_length, num_labels].

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    num_labels = 2

    model = BertForTokenClassification(config, num_labels)
    logits = model(input_ids, token_type_ids, input_mask)
    ```
    """

    def __init__(self, config, num_labels=2):
        super(BertForTokenClassification, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        sequence_output, _ = self.bert(
            input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(
                    logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits


class BertForQuestionAnswering(PreTrainedBertModel):
    """BERT model for Question Answering (span extraction).
    This module is composed of the BERT model with a linear layer on top of
    the sequence output that computes start_logits and end_logits

    Params:
        `config`: either
            - a BertConfig class instance with the configuration to build a new model, or
            - a str with the name of a pre-trained model to load selected in the list of:
                . `bert-base-uncased`
                . `bert-large-uncased`
                . `bert-base-cased`
                . `bert-base-multilingual`
                . `bert-base-chinese`
                The pre-trained model will be downloaded and cached if needed.

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `start_positions`: position of the first token for the labeled span: torch.LongTensor of shape [batch_size].
            Positions are clamped to the length of the sequence and position outside of the sequence are not taken
            into account for computing the loss.
        `end_positions`: position of the last token for the labeled span: torch.LongTensor of shape [batch_size].
            Positions are clamped to the length of the sequence and position outside of the sequence are not taken
            into account for computing the loss.

    Outputs:
        if `start_positions` and `end_positions` are not `None`:
            Outputs the total_loss which is the sum of the CrossEntropy loss for the start and end token positions.
        if `start_positions` or `end_positions` is `None`:
            Outputs a tuple of start_logits, end_logits which are the logits respectively for the start and end
            position tokens of shape [batch_size, sequence_length].

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    model = BertForQuestionAnswering(config)
    start_logits, end_logits = model(input_ids, token_type_ids, input_mask)
    ```
    """

    def __init__(self, config):
        super(BertForQuestionAnswering, self).__init__(config)
        self.bert = BertModel(config)
        # TODO check with Google if it's normal there is no dropout on the token classifier of SQuAD in the TF version
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.qa_outputs = nn.Linear(config.hidden_size, 2)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, start_positions=None, end_positions=None):
        sequence_output, _ = self.bert(
            input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            return total_loss
        else:
            return start_logits, end_logits

class BertForSignOnly(PreTrainedBertModel):
    """refer to BertForPreTraining"""

    def __init__(self, config, num_labels=2, enable_butd=False, len_vis_input=36,
                 visdial_v='1.0', loss_type='mlm', eval_disc=False, float_nsp_label=False,
                 neg_num=0, adaptive_weight=False, add_attn_fuse=False,
                 no_h0=False, add_val=False, no_vision=False, rank_loss='',**kwargs):
        super(BertForSignOnly, self).__init__(config)
        self.bert = BertModel(config)
        self.cls = BertPreTrainingHeads(
            config, self.bert.embeddings.word_embeddings.weight,
            num_labels=num_labels)  # num_labels not applicable for VLP
        self.apply(self.init_bert_weights)
        self.crit_mask_lm = nn.CrossEntropyLoss(reduction='none')
        self.num_labels = num_labels
        self.len_vis_input = len_vis_input
        self.enable_butd = enable_butd
        self.eval_disc = eval_disc
        self.loss_type = loss_type
        self.visdial_v = visdial_v
        self.float_nsp_label = float_nsp_label
        self.add_attn_fuse = add_attn_fuse
        self.no_h0 = no_h0
        self.add_val = add_val
        self.no_vision = no_vision
        self.rank_loss = rank_loss
        self.dropout = nn.Dropout(config.hidden_dropout_prob)  # for self_attn

        if hasattr(config, 'label_smoothing') and config.label_smoothing:
            self.crit_mask_lm_smoothed = LabelSmoothingLoss(
                config.label_smoothing, config.vocab_size, ignore_index=0, reduction='none')
        else:
            self.crit_mask_lm_smoothed = None

        if 'nsp' in loss_type:
            if self.float_nsp_label:
                if self.rank_loss == 'softmax':
                    self.ce_loss_fct = nn.KLDivLoss(reduction='batchmean')
                else:
                    self.crit_next_sent = nn.CrossEntropyLoss(ignore_index=-1, reduction='none')
            else:
                if adaptive_weight and neg_num > 1:
                    weight = [2.0 / (1.0 + neg_num), 2 * neg_num / (1.0 + neg_num)]
                    print("Setting adaptive weights for neg/pos=%.2f/%.2f" % (weight[0], weight[1]))
                    weight = torch.tensor(weight, dtype=torch.float32).cuda()
                    self.crit_next_sent = nn.CrossEntropyLoss(ignore_index=-1, weight=weight)

                else:
                    self.crit_next_sent = nn.CrossEntropyLoss(ignore_index=-1)
        self.logsoftmax = nn.LogSoftmax(dim=-1)

        if enable_butd:
            if len_vis_input == 36:
                self.vis_embed = nn.Sequential(nn.Linear(2048, config.hidden_size),
                                               nn.ReLU(),
                                               nn.Dropout(config.hidden_dropout_prob))  # use to be 0.3
                self.vis_pe_embed = nn.Sequential(nn.Linear(7, config.hidden_size),
                                                  nn.ReLU(),
                                                  nn.Dropout(config.hidden_dropout_prob))
            elif visdial_v == "0.9" and len_vis_input == 100:
                self.vis_embed = nn.Sequential(nn.Linear(2048, 2048),
                                               nn.ReLU(),
                                               nn.Linear(2048, config.hidden_size),
                                               nn.ReLU(),
                                               nn.Dropout(config.hidden_dropout_prob))  # use to be 0.3
                try:
                    self.vis_embed[0].weight.data.copy_(torch.from_numpy(pickle.load(
                        open('detectron_weights/fc7_w.pkl', 'rb'))))
                    self.vis_embed[0].bias.data.copy_(torch.from_numpy(pickle.load(
                        open('detectron_weights/fc7_b.pkl', 'rb'))))
                except:
                    raise Exception(
                        'Cannot find Detectron fc7 weights! Download from https://dl.fbaipublicfiles.com/ActivityNet-Entities/ActivityNet-Entities/detectron_weights.tar.gz and uncompress under the code root directory.')

                self.vis_pe_embed = nn.Sequential(nn.Linear(6 + 1601, config.hidden_size),
                                                  nn.ReLU(),
                                                  nn.Dropout(config.hidden_dropout_prob))

    def forward(self, vis_feats, vis_pe, input_ids, token_type_ids=None, attention_mask=None, masked_lm_labels=None,
                next_sentence_label=None, masked_pos=None, masked_weights=None, task_idx=None,
                vis_masked_pos=[], mask_image_regions=False, drop_worst_ratio=0.2,len_vis_input = 36):

        
        def gather_seq_out_by_pos(seq, pos):
            return torch.gather(seq, 1, pos.unsqueeze(2).expand(-1, -1, seq.size(-1)))
   
        sequence_output, pooled_output = self.bert(vis_feats, vis_pe, input_ids, token_type_ids,
                                                       attention_mask, output_all_encoded_layers=False,
                                                       len_vis_input=len_vis_input)

        def loss_mask_and_normalize(loss, mask, drop_worst_ratio):
            mask = mask.type_as(loss)
            loss = loss * mask

            # Ruotian Luo's drop worst
            keep_loss, keep_ind = torch.topk(loss.sum(-1), int(loss.size(0) * (1 - drop_worst_ratio)), largest=False)

            # denominator = torch.sum(mask) + 1e-5
            # return (loss / denominator).sum()
            denominator = torch.sum(mask.sum(-1)[keep_ind]) + 1e-5
            return (keep_loss / denominator).sum()

        # masked lm
        if self.loss_type == 'ctc':
            return sequence_output
        elif self.loss_type == 'mlm':
            if masked_pos.numel() == 0:
                # hack to avoid empty masked_pos during training for now
                masked_lm_loss = pooled_output.new(1).fill_(0)
            else:
                sequence_output_masked = gather_seq_out_by_pos(
                    sequence_output, masked_pos)
                prediction_scores_masked, seq_relationship_score = self.cls(
                    sequence_output_masked, pooled_output, task_idx=task_idx)
                if self.crit_mask_lm_smoothed:
                    masked_lm_loss = self.crit_mask_lm_smoothed(
                        F.log_softmax(prediction_scores_masked.float(), dim=-1), masked_lm_labels)
                else:
                    masked_lm_loss = self.crit_mask_lm(
                        prediction_scores_masked.transpose(1, 2).float(), masked_lm_labels)
                masked_lm_loss = loss_mask_and_normalize(
                    masked_lm_loss.float(), masked_weights, drop_worst_ratio)
            next_sentence_loss = masked_lm_loss.new(1).fill_(0)
            
        elif self.loss_type == 'nsp':
            prediction_scores, seq_relationship_score = self.cls(
                sequence_output, pooled_output, task_idx=task_idx)
            if self.float_nsp_label:
                rel_scores = next_sentence_label.view(-1)
                binary_label = rel_scores.ceil().long()
                zero_tensor = seq_relationship_score.new(1).fill_(0)
                if self.rank_loss:
                    output = seq_relationship_score.view(-1, 30, 2)[:, :, -1]
                    rs_score = rel_scores.view(-1, 30)
                    if self.rank_loss == 'softmax':
                        next_sentence_loss = self.ce_loss_fct(F.log_softmax(output, dim=1), F.softmax(rs_score, dim=1))
                    elif self.rank_loss == 'listmle':
                        next_sentence_loss = listMLE(output, rs_score)
                    elif self.rank_loss == 'listnet':
                        next_sentence_loss = listNet(output, rs_score)
                    elif self.rank_loss == 'approxndcg':
                        next_sentence_loss = approxNDCGLoss(output, rs_score)
                    else:
                        raise NotImplementedError
                    return zero_tensor, zero_tensor, next_sentence_loss

                if self.add_val:
                    loss_weights = torch.where(rel_scores == 0, torch.ones_like(rel_scores), rel_scores * 1)
                else:
                    loss_weights = torch.where(rel_scores == 0, torch.ones_like(rel_scores), rel_scores * 2)
                next_sentence_losses = self.crit_next_sent(
                    seq_relationship_score.view(-1, self.num_labels).float(), binary_label)
                next_sentence_loss = torch.mean(next_sentence_losses * loss_weights)
            else:
                next_sentence_loss = self.crit_next_sent(
                    seq_relationship_score.view(-1, self.num_labels).float(), next_sentence_label.view(-1))
            masked_lm_loss = next_sentence_loss.new(1).fill_(0)
        elif self.loss_type == 'mlm_nsp':
            if masked_pos.numel() == 0:
                # hack to avoid empty masked_pos during training for now
                masked_lm_loss = pooled_output.new(1).fill_(0)
            else:
                sequence_output_masked = gather_seq_out_by_pos(
                    sequence_output, masked_pos)
                prediction_scores_masked, seq_relationship_score = self.cls(
                    sequence_output_masked, pooled_output, task_idx=task_idx)
                if self.crit_mask_lm_smoothed:
                    masked_lm_loss = self.crit_mask_lm_smoothed(
                        F.log_softmax(prediction_scores_masked.float(), dim=-1), masked_lm_labels)
                else:
                    masked_lm_loss = self.crit_mask_lm(
                        prediction_scores_masked.transpose(1, 2).float(), masked_lm_labels)
                masked_lm_loss = loss_mask_and_normalize(
                    masked_lm_loss.float(), masked_weights, drop_worst_ratio)
            next_sentence_loss = self.crit_next_sent(
                seq_relationship_score.view(-1, self.num_labels).float(), next_sentence_label.view(-1))
        else:
            raise NotImplementedError

        if mask_image_regions:
            # Selfie-like pretext
            masked_vis_feats = torch.gather(vis_feats, 1,
                                            (vis_masked_pos - 1).unsqueeze(-1).expand((-1, -1, vis_feats.size(-1))))

            if self.enable_butd:
                masked_pos_enc = torch.gather(vis_pe, 1,
                                              (vis_masked_pos - 1).unsqueeze(-1).expand((-1, -1, vis_pe.size(-1))))
            else:
                masked_pos_enc = self.bert.embeddings.position_embeddings(vis_masked_pos)

            masked_pos_enc += pooled_output.unsqueeze(1).expand_as(masked_pos_enc)
            assert (masked_vis_feats.size() == masked_pos_enc.size())
            sim_mat = torch.matmul(masked_pos_enc, masked_vis_feats.permute(0, 2, 1).contiguous())
            sim_mat = F.log_softmax(sim_mat, dim=-1)
            vis_pretext_loss = []
            for i in range(sim_mat.size(0)):
                vis_pretext_loss.append(sim_mat[i].diag().mean().view(1) * -1.)  # cross entropy for ones
            vis_pretext_loss = torch.cat(vis_pretext_loss).mean()
        else:
            vis_pretext_loss = masked_lm_loss.new(1).fill_(0)

        return masked_lm_loss, vis_pretext_loss, next_sentence_loss

class BertMultiScaleEmbedding(PreTrainedBertModel):
    """refer to BertForPreTraining"""

    def __init__(self, config, num_labels=2, enable_butd=False, len_vis_input=36,
                 visdial_v='1.0', loss_type='mlm', eval_disc=False, float_nsp_label=False,
                 neg_num=0, adaptive_weight=False, add_attn_fuse=False,
                 no_h0=False, add_val=False, no_vision=False, rank_loss='',**kwargs):
        super(BertMultiScaleEmbedding, self).__init__(config)
        self.bert = BertMultiScaleEmbeddingModel(config)
        self.cls = BertPreTrainingHeads(
            config, self.bert.embeddings.word_embeddings.weight,
            num_labels=num_labels)  # num_labels not applicable for VLP
        self.apply(self.init_bert_weights)
        self.crit_mask_lm = nn.CrossEntropyLoss(reduction='none')
        self.num_labels = num_labels
        self.len_vis_input = len_vis_input
        self.enable_butd = enable_butd
        self.eval_disc = eval_disc
        self.loss_type = loss_type
        self.visdial_v = visdial_v
        self.float_nsp_label = float_nsp_label
        self.add_attn_fuse = add_attn_fuse
        self.no_h0 = no_h0
        self.add_val = add_val
        self.no_vision = no_vision
        self.rank_loss = rank_loss
        self.dropout = nn.Dropout(config.hidden_dropout_prob)  # for self_attn

        if hasattr(config, 'label_smoothing') and config.label_smoothing:
            self.crit_mask_lm_smoothed = LabelSmoothingLoss(
                config.label_smoothing, config.vocab_size, ignore_index=0, reduction='none')
        else:
            self.crit_mask_lm_smoothed = None

        if 'nsp' in loss_type:
            if self.float_nsp_label:
                if self.rank_loss == 'softmax':
                    self.ce_loss_fct = nn.KLDivLoss(reduction='batchmean')
                else:
                    self.crit_next_sent = nn.CrossEntropyLoss(ignore_index=-1, reduction='none')
            else:
                if adaptive_weight and neg_num > 1:
                    weight = [2.0 / (1.0 + neg_num), 2 * neg_num / (1.0 + neg_num)]
                    print("Setting adaptive weights for neg/pos=%.2f/%.2f" % (weight[0], weight[1]))
                    weight = torch.tensor(weight, dtype=torch.float32).cuda()
                    self.crit_next_sent = nn.CrossEntropyLoss(ignore_index=-1, weight=weight)

                else:
                    self.crit_next_sent = nn.CrossEntropyLoss(ignore_index=-1)
        self.logsoftmax = nn.LogSoftmax(dim=-1)

        if enable_butd:
            if len_vis_input == 36:
                self.vis_embed = nn.Sequential(nn.Linear(2048, config.hidden_size),
                                               nn.ReLU(),
                                               nn.Dropout(config.hidden_dropout_prob))  # use to be 0.3
                self.vis_pe_embed = nn.Sequential(nn.Linear(7, config.hidden_size),
                                                  nn.ReLU(),
                                                  nn.Dropout(config.hidden_dropout_prob))
            elif visdial_v == "0.9" and len_vis_input == 100:
                self.vis_embed = nn.Sequential(nn.Linear(2048, 2048),
                                               nn.ReLU(),
                                               nn.Linear(2048, config.hidden_size),
                                               nn.ReLU(),
                                               nn.Dropout(config.hidden_dropout_prob))  # use to be 0.3
                try:
                    self.vis_embed[0].weight.data.copy_(torch.from_numpy(pickle.load(
                        open('detectron_weights/fc7_w.pkl', 'rb'))))
                    self.vis_embed[0].bias.data.copy_(torch.from_numpy(pickle.load(
                        open('detectron_weights/fc7_b.pkl', 'rb'))))
                except:
                    raise Exception(
                        'Cannot find Detectron fc7 weights! Download from https://dl.fbaipublicfiles.com/ActivityNet-Entities/ActivityNet-Entities/detectron_weights.tar.gz and uncompress under the code root directory.')

                self.vis_pe_embed = nn.Sequential(nn.Linear(6 + 1601, config.hidden_size),
                                                  nn.ReLU(),
                                                  nn.Dropout(config.hidden_dropout_prob))

    def forward(self, vis_feats, vis_pe, input_ids, token_type_ids=None, attention_mask=None, attention_mask_coarse=None,masked_lm_labels=None,
                next_sentence_label=None, masked_pos=None, masked_weights=None, task_idx=None,
                vis_masked_pos=[], mask_image_regions=False, drop_worst_ratio=0.2,len_vis_input = 36):

        
        def gather_seq_out_by_pos(seq, pos):
            return torch.gather(seq, 1, pos.unsqueeze(2).expand(-1, -1, seq.size(-1)))
   
        sequence_output_fine, sequence_output_coarse = self.bert(vis_feats, vis_pe, input_ids, token_type_ids,
                                                       attention_mask,attention_mask_coarse, output_all_encoded_layers=False,
                                                       len_vis_input=len_vis_input) 
        return sequence_output_fine,sequence_output_coarse
        
class BertTextEmbedding(PreTrainedBertModel):
    """refer to BertForPreTraining"""

    def __init__(self, config, num_labels=2, enable_butd=False, len_vis_input=36,
                 visdial_v='1.0', loss_type='mlm', eval_disc=False, float_nsp_label=False,
                 neg_num=0, adaptive_weight=False, add_attn_fuse=False,
                 no_h0=False, add_val=False, no_vision=False, rank_loss='',**kwargs):
        super(BertTextEmbedding, self).__init__(config)
        # self.bert = BertModel(config)
        self.bert = BertEmbeddings(config)
        self.cls = BertPreTrainingHeads(
            config, self.bert.word_embeddings.weight,
            num_labels=num_labels)  # num_labels not applicable for VLP
        self.apply(self.init_bert_weights)
        self.crit_mask_lm = nn.CrossEntropyLoss(reduction='none')
        self.num_labels = num_labels
        self.len_vis_input = len_vis_input
        self.enable_butd = enable_butd
        self.eval_disc = eval_disc
        self.loss_type = loss_type
        self.visdial_v = visdial_v
        self.float_nsp_label = float_nsp_label
        self.add_attn_fuse = add_attn_fuse
        self.no_h0 = no_h0
        self.add_val = add_val
        self.no_vision = no_vision
        self.rank_loss = rank_loss
        self.dropout = nn.Dropout(config.hidden_dropout_prob)  # for self_attn

        if hasattr(config, 'label_smoothing') and config.label_smoothing:
            self.crit_mask_lm_smoothed = LabelSmoothingLoss(
                config.label_smoothing, config.vocab_size, ignore_index=0, reduction='none')
        else:
            self.crit_mask_lm_smoothed = None

        if 'nsp' in loss_type:
            if self.float_nsp_label:
                if self.rank_loss == 'softmax':
                    self.ce_loss_fct = nn.KLDivLoss(reduction='batchmean')
                else:
                    self.crit_next_sent = nn.CrossEntropyLoss(ignore_index=-1, reduction='none')
            else:
                if adaptive_weight and neg_num > 1:
                    weight = [2.0 / (1.0 + neg_num), 2 * neg_num / (1.0 + neg_num)]
                    print("Setting adaptive weights for neg/pos=%.2f/%.2f" % (weight[0], weight[1]))
                    weight = torch.tensor(weight, dtype=torch.float32).cuda()
                    self.crit_next_sent = nn.CrossEntropyLoss(ignore_index=-1, weight=weight)

                else:
                    self.crit_next_sent = nn.CrossEntropyLoss(ignore_index=-1)
        self.logsoftmax = nn.LogSoftmax(dim=-1)

        if enable_butd:
            if len_vis_input == 36:
                self.vis_embed = nn.Sequential(nn.Linear(2048, config.hidden_size),
                                               nn.ReLU(),
                                               nn.Dropout(config.hidden_dropout_prob))  # use to be 0.3
                self.vis_pe_embed = nn.Sequential(nn.Linear(7, config.hidden_size),
                                                  nn.ReLU(),
                                                  nn.Dropout(config.hidden_dropout_prob))
            elif visdial_v == "0.9" and len_vis_input == 100:
                self.vis_embed = nn.Sequential(nn.Linear(2048, 2048),
                                               nn.ReLU(),
                                               nn.Linear(2048, config.hidden_size),
                                               nn.ReLU(),
                                               nn.Dropout(config.hidden_dropout_prob))  # use to be 0.3
                try:
                    self.vis_embed[0].weight.data.copy_(torch.from_numpy(pickle.load(
                        open('detectron_weights/fc7_w.pkl', 'rb'))))
                    self.vis_embed[0].bias.data.copy_(torch.from_numpy(pickle.load(
                        open('detectron_weights/fc7_b.pkl', 'rb'))))
                except:
                    raise Exception(
                        'Cannot find Detectron fc7 weights! Download from https://dl.fbaipublicfiles.com/ActivityNet-Entities/ActivityNet-Entities/detectron_weights.tar.gz and uncompress under the code root directory.')

                self.vis_pe_embed = nn.Sequential(nn.Linear(6 + 1601, config.hidden_size),
                                                  nn.ReLU(),
                                                  nn.Dropout(config.hidden_dropout_prob))

    def forward(self, vis_feats, vis_pe, input_ids, token_type_ids=None, attention_mask=None, masked_lm_labels=None,
                next_sentence_label=None, masked_pos=None, masked_weights=None, task_idx=None,
                vis_masked_pos=[], mask_image_regions=False, drop_worst_ratio=0.2,len_vis_input = 36):

        
        def gather_seq_out_by_pos(seq, pos):
            return torch.gather(seq, 1, pos.unsqueeze(2).expand(-1, -1, seq.size(-1)))
   
        sequence_output= self.bert(vis_feats, vis_pe, input_ids, token_type_ids,
                                                      
                                                       len_vis_input=len_vis_input)

        def loss_mask_and_normalize(loss, mask, drop_worst_ratio):
            mask = mask.type_as(loss)
            loss = loss * mask

            # Ruotian Luo's drop worst
            keep_loss, keep_ind = torch.topk(loss.sum(-1), int(loss.size(0) * (1 - drop_worst_ratio)), largest=False)

            # denominator = torch.sum(mask) + 1e-5
            # return (loss / denominator).sum()
            denominator = torch.sum(mask.sum(-1)[keep_ind]) + 1e-5
            return (keep_loss / denominator).sum()

        return sequence_output



class BertNoEmbedding(PreTrainedBertModel):
    """refer to BertForPreTraining"""

    def __init__(self, config, num_labels=2, enable_butd=False, len_vis_input=36,
                 visdial_v='1.0', loss_type='mlm', eval_disc=False, float_nsp_label=False,
                 neg_num=0, adaptive_weight=False, add_attn_fuse=False,
                 no_h0=False, add_val=False, no_vision=False, rank_loss='',**kwargs):
        super(BertNoEmbedding, self).__init__(config)
        self.bert = BertModelAfterEmbedding(config)
        self.cls = BertPreTrainingHeads(
            config, self.bert.embeddings.word_embeddings.weight,
            num_labels=num_labels)  # num_labels not applicable for VLP
        self.apply(self.init_bert_weights)
        self.crit_mask_lm = nn.CrossEntropyLoss(reduction='none')
        self.num_labels = num_labels
        self.len_vis_input = len_vis_input
        self.enable_butd = enable_butd
        self.eval_disc = eval_disc
        self.loss_type = loss_type
        self.visdial_v = visdial_v
        self.float_nsp_label = float_nsp_label
        self.add_attn_fuse = add_attn_fuse
        self.no_h0 = no_h0
        self.add_val = add_val
        self.no_vision = no_vision
        self.rank_loss = rank_loss
        self.dropout = nn.Dropout(config.hidden_dropout_prob)  # for self_attn

        if hasattr(config, 'label_smoothing') and config.label_smoothing:
            self.crit_mask_lm_smoothed = LabelSmoothingLoss(
                config.label_smoothing, config.vocab_size, ignore_index=0, reduction='none')
        else:
            self.crit_mask_lm_smoothed = None

        if 'nsp' in loss_type:
            if self.float_nsp_label:
                if self.rank_loss == 'softmax':
                    self.ce_loss_fct = nn.KLDivLoss(reduction='batchmean')
                else:
                    self.crit_next_sent = nn.CrossEntropyLoss(ignore_index=-1, reduction='none')
            else:
                if adaptive_weight and neg_num > 1:
                    weight = [2.0 / (1.0 + neg_num), 2 * neg_num / (1.0 + neg_num)]
                    print("Setting adaptive weights for neg/pos=%.2f/%.2f" % (weight[0], weight[1]))
                    weight = torch.tensor(weight, dtype=torch.float32).cuda()
                    self.crit_next_sent = nn.CrossEntropyLoss(ignore_index=-1, weight=weight)

                else:
                    self.crit_next_sent = nn.CrossEntropyLoss(ignore_index=-1)
        self.logsoftmax = nn.LogSoftmax(dim=-1)


    def forward(self, input_embedding, attention_mask=None, masked_lm_labels=None,
                next_sentence_label=None, masked_pos=None, masked_weights=None, task_idx=None,
                vis_masked_pos=[], mask_image_regions=False, drop_worst_ratio=0.2,len_vis_input = 36):

        
        def gather_seq_out_by_pos(seq, pos):
            return torch.gather(seq, 1, pos.unsqueeze(2).expand(-1, -1, seq.size(-1)))
   
        sequence_output, pooled_output = self.bert(input_embedding,attention_mask, 
                                                    output_all_encoded_layers=False
                                                    )
        
        # print(sequence_output.size())

        def loss_mask_and_normalize(loss, mask, drop_worst_ratio):
            mask = mask.type_as(loss)
            loss = loss * mask

            # Ruotian Luo's drop worst
            keep_loss, keep_ind = torch.topk(loss.sum(-1), int(loss.size(0) * (1 - drop_worst_ratio)), largest=False)

            # denominator = torch.sum(mask) + 1e-5
            # return (loss / denominator).sum()
            denominator = torch.sum(mask.sum(-1)[keep_ind]) + 1e-5
            return (keep_loss / denominator).sum()

        # masked lm
        if self.loss_type == 'ctc':
            return sequence_output
        elif self.loss_type == 'ctc_nsp':
                
            prediction_scores, seq_relationship_score = self.cls(
                sequence_output, pooled_output, task_idx=task_idx)
            # 测试时
            if next_sentence_label == None:
                return sequence_output,seq_relationship_score
            next_sentence_loss = self.crit_next_sent(
                    seq_relationship_score.view(-1, self.num_labels).float(), next_sentence_label.view(-1))
            masked_lm_loss = next_sentence_loss.new(1).fill_(0)
            
            return sequence_output,next_sentence_loss,seq_relationship_score


class BertSignFusion(PreTrainedBertModel):
    def __init__(self, config, num_labels=2, enable_butd=False, len_vis_input=36,
                 visdial_v='1.0', loss_type='mlm', eval_disc=False, float_nsp_label=False,
                 neg_num=0, adaptive_weight=False, add_attn_fuse=False,
                 no_h0=False, add_val=False, no_vision=False, rank_loss='',**kwargs):
        super(BertSignFusion, self).__init__(config)
        self.bert = BertModelFusion(config)
        self.cls = BertPreTrainingHeads(
            config, self.bert.embeddings.word_embeddings.weight,
            num_labels=num_labels)  # num_labels not applicable for VLP
        self.apply(self.init_bert_weights)
        self.crit_mask_lm = nn.CrossEntropyLoss(reduction='none')
        self.num_labels = num_labels
        self.len_vis_input = len_vis_input
        self.enable_butd = enable_butd
        self.eval_disc = eval_disc
        self.loss_type = loss_type
        self.visdial_v = visdial_v
        self.float_nsp_label = float_nsp_label
        self.add_attn_fuse = add_attn_fuse
        self.no_h0 = no_h0
        self.add_val = add_val
        self.no_vision = no_vision
        self.rank_loss = rank_loss
        self.dropout = nn.Dropout(config.hidden_dropout_prob)  # for self_attn

        if hasattr(config, 'label_smoothing') and config.label_smoothing:
            self.crit_mask_lm_smoothed = LabelSmoothingLoss(
                config.label_smoothing, config.vocab_size, ignore_index=0, reduction='none')
        else:
            self.crit_mask_lm_smoothed = None

        if 'nsp' in loss_type:
            if self.float_nsp_label:
                if self.rank_loss == 'softmax':
                    self.ce_loss_fct = nn.KLDivLoss(reduction='batchmean')
                else:
                    self.crit_next_sent = nn.CrossEntropyLoss(ignore_index=-1, reduction='none')
            else:
                if adaptive_weight and neg_num > 1:
                    weight = [2.0 / (1.0 + neg_num), 2 * neg_num / (1.0 + neg_num)]
                    print("Setting adaptive weights for neg/pos=%.2f/%.2f" % (weight[0], weight[1]))
                    weight = torch.tensor(weight, dtype=torch.float32).cuda()
                    self.crit_next_sent = nn.CrossEntropyLoss(ignore_index=-1, weight=weight)

                else:
                    self.crit_next_sent = nn.CrossEntropyLoss(ignore_index=-1)
        self.logsoftmax = nn.LogSoftmax(dim=-1)


    def forward(self, vis_feat,text_feat,position_ids, attention_mask=None, masked_lm_labels=None,
                next_sentence_label=None, masked_pos=None, masked_weights=None, task_idx=None,
                vis_masked_pos=[], mask_image_regions=False, drop_worst_ratio=0.2,len_vis_input = 36):

        
        def gather_seq_out_by_pos(seq, pos):
            return torch.gather(seq, 1, pos.unsqueeze(2).expand(-1, -1, seq.size(-1)))
   
        sequence_output, pooled_output = self.bert(vis_feat,text_feat,position_ids,attention_mask, 
                                                    output_all_encoded_layers=False
                                                    )
        
        # print(sequence_output.size())
        # vd-bert中bs为32，使用了drop worst，slr的bs只有2，所以不用
        def loss_mask_and_normalize(loss, mask, drop_worst_ratio):
            mask = mask.type_as(loss)
            loss = loss * mask

            return loss.sum() / mask.sum()

        # def loss_mask_and_normalize(loss, mask, drop_worst_ratio):
        #     mask = mask.type_as(loss)
        #     loss = loss * mask

        #     # Ruotian Luo's drop worst
        #     keep_loss, keep_ind = torch.topk(loss.sum(-1), int(loss.size(0) * (1 - drop_worst_ratio)), largest=False)

        #     # denominator = torch.sum(mask) + 1e-5
        #     # return (loss / denominator).sum()
        #     denominator = torch.sum(mask.sum(-1)[keep_ind]) + 1e-5
        #     return (keep_loss / denominator).sum()

        # masked lm
        if self.loss_type == 'ctc':
            return sequence_output
        elif self.loss_type == 'ctc_nsp':
                
            prediction_scores, seq_relationship_score = self.cls(
                sequence_output, pooled_output, task_idx=task_idx)
            # 测试时
            if next_sentence_label == None:
                return sequence_output,seq_relationship_score
            next_sentence_loss = self.crit_next_sent(
                    seq_relationship_score.view(-1, self.num_labels).float(), next_sentence_label.view(-1))
            masked_lm_loss = next_sentence_loss.new(1).fill_(0)
            
            return sequence_output,next_sentence_loss,seq_relationship_score
        elif self.loss_type == 'ctc_nsp_mlm':
            if masked_lm_labels == None or masked_pos.numel() == 0:
                # hack to avoid empty masked_pos during training for now
                masked_lm_loss = pooled_output.new(1).fill_(0)
                prediction_scores, seq_relationship_score = self.cls(
                    sequence_output, pooled_output, task_idx=task_idx)
                
            else:
                sequence_output_masked = gather_seq_out_by_pos(
                    sequence_output, masked_pos)
                prediction_scores_masked, seq_relationship_score = self.cls(
                    sequence_output_masked, pooled_output, task_idx=task_idx)
                if self.crit_mask_lm_smoothed:
                    masked_lm_loss = self.crit_mask_lm_smoothed(
                        F.log_softmax(prediction_scores_masked.float(), dim=-1), masked_lm_labels)
                else:
                    masked_lm_loss = self.crit_mask_lm(
                        prediction_scores_masked.transpose(1, 2).float(), masked_lm_labels)
                
                
                masked_lm_loss = loss_mask_and_normalize(
                    masked_lm_loss.float(), masked_weights, drop_worst_ratio)
            
            if next_sentence_label == None or masked_lm_labels == None:
                return sequence_output,seq_relationship_score
                
            # prediction_scores, seq_relationship_score = self.cls(
            #     sequence_output, pooled_output, task_idx=task_idx)
            next_sentence_loss = self.crit_next_sent(
                    seq_relationship_score.view(-1, self.num_labels).float(), next_sentence_label.view(-1))
            
            
            return sequence_output,next_sentence_loss,seq_relationship_score,masked_lm_loss


class BertSignTokenization(PreTrainedBertModel):
    """refer to BertForPreTraining"""

    def __init__(self, config, num_labels=2, enable_butd=False, len_vis_input=36,
                 visdial_v='1.0', loss_type='mlm', eval_disc=False, float_nsp_label=False,
                 neg_num=0, adaptive_weight=False, add_attn_fuse=False,
                 no_h0=False, add_val=False, no_vision=False, rank_loss='',**kwargs):
        super(BertSignTokenization, self).__init__(config)
        self.bert = BertModelSignTokenization(config)
        self.cls = BertPreTrainingHeads(
            config, self.bert.embeddings.word_embeddings.weight,
            num_labels=num_labels)  # num_labels not applicable for VLP
        self.apply(self.init_bert_weights)
        self.crit_mask_lm = nn.CrossEntropyLoss(reduction='none')
        self.num_labels = num_labels
        self.len_vis_input = len_vis_input
        self.enable_butd = enable_butd
        self.eval_disc = eval_disc
        self.loss_type = loss_type
        self.visdial_v = visdial_v
        self.float_nsp_label = float_nsp_label
        self.add_attn_fuse = add_attn_fuse
        self.no_h0 = no_h0
        self.add_val = add_val
        self.no_vision = no_vision
        self.rank_loss = rank_loss
        self.dropout = nn.Dropout(config.hidden_dropout_prob)  # for self_attn

        if hasattr(config, 'label_smoothing') and config.label_smoothing:
            self.crit_mask_lm_smoothed = LabelSmoothingLoss(
                config.label_smoothing, config.vocab_size, ignore_index=0, reduction='none')
        else:
            self.crit_mask_lm_smoothed = None
 

    def forward(self, vis_feats,vis_pe,token_type_ids,attention_mask):

        sequence_output, pooled_output = self.bert(vis_feats,vis_pe,token_type_ids,attention_mask, 
                                                    output_all_encoded_layers=False)
        return sequence_output



class BertHybridFusion(PreTrainedBertModel):
    def __init__(self, config, num_labels=2, enable_butd=False, len_vis_input=36,
                 visdial_v='1.0', loss_type='mlm', eval_disc=False, float_nsp_label=False,
                 neg_num=0, adaptive_weight=False, add_attn_fuse=False,
                 no_h0=False, add_val=False, no_vision=False, rank_loss='',**kwargs):
        super(BertHybridFusion, self).__init__(config)
        self.bert = BertModelHybrid(config)
        self.cls = BertPreTrainingHeads(
            config, self.bert.embeddings.word_embeddings.weight,
            num_labels=num_labels)  # num_labels not applicable for VLP
        self.apply(self.init_bert_weights)
        self.crit_mask_lm = nn.CrossEntropyLoss(reduction='none')
        self.num_labels = num_labels
        self.len_vis_input = len_vis_input
        self.enable_butd = enable_butd
        self.eval_disc = eval_disc
        self.loss_type = loss_type
        self.visdial_v = visdial_v
        self.float_nsp_label = float_nsp_label
        self.add_attn_fuse = add_attn_fuse
        self.no_h0 = no_h0
        self.add_val = add_val
        self.no_vision = no_vision
        self.rank_loss = rank_loss
        self.dropout = nn.Dropout(config.hidden_dropout_prob)  # for self_attn

        if hasattr(config, 'label_smoothing') and config.label_smoothing:
            self.crit_mask_lm_smoothed = LabelSmoothingLoss(
                config.label_smoothing, config.vocab_size, ignore_index=0, reduction='none')
        else:
            self.crit_mask_lm_smoothed = None

        if 'nsp' in loss_type:
            if self.float_nsp_label:
                if self.rank_loss == 'softmax':
                    self.ce_loss_fct = nn.KLDivLoss(reduction='batchmean')
                else:
                    self.crit_next_sent = nn.CrossEntropyLoss(ignore_index=-1, reduction='none')
            else:
                if adaptive_weight and neg_num > 1:
                    weight = [2.0 / (1.0 + neg_num), 2 * neg_num / (1.0 + neg_num)]
                    print("Setting adaptive weights for neg/pos=%.2f/%.2f" % (weight[0], weight[1]))
                    weight = torch.tensor(weight, dtype=torch.float32).cuda()
                    self.crit_next_sent = nn.CrossEntropyLoss(ignore_index=-1, weight=weight)

                else:
                    self.crit_next_sent = nn.CrossEntropyLoss(ignore_index=-1)
        self.logsoftmax = nn.LogSoftmax(dim=-1)


    def forward(self, vis_feat,text_feat,position_ids, attention_mask_vis, attention_mask_text, attention_mask_VasQ, masked_lm_labels=None,
                next_sentence_label=None, masked_pos=None, masked_weights=None, task_idx=None,
                vis_masked_pos=[], mask_image_regions=False, drop_worst_ratio=0.2,len_vis_input = 36):

        
        def gather_seq_out_by_pos(seq, pos):
            return torch.gather(seq, 1, pos.unsqueeze(2).expand(-1, -1, seq.size(-1)))
   
        sequence_output, pooled_output = self.bert(
            vis_feat, text_feat, position_ids, 
            attention_mask_vis, attention_mask_text, attention_mask_VasQ,
            output_all_encoded_layers=False
        )
        
        # print(sequence_output.size())
        # vd-bert中bs为32，使用了drop worst，slr的bs只有2，所以不用
        def loss_mask_and_normalize(loss, mask, drop_worst_ratio):
            mask = mask.type_as(loss)
            loss = loss * mask

            return loss.sum() / mask.sum()

        # def loss_mask_and_normalize(loss, mask, drop_worst_ratio):
        #     mask = mask.type_as(loss)
        #     loss = loss * mask

        #     # Ruotian Luo's drop worst
        #     keep_loss, keep_ind = torch.topk(loss.sum(-1), int(loss.size(0) * (1 - drop_worst_ratio)), largest=False)

        #     # denominator = torch.sum(mask) + 1e-5
        #     # return (loss / denominator).sum()
        #     denominator = torch.sum(mask.sum(-1)[keep_ind]) + 1e-5
        #     return (keep_loss / denominator).sum()

        # masked lm
        if self.loss_type == 'ctc':
            return sequence_output
        elif self.loss_type == 'ctc_nsp':
            # sequence_output[0] 是视频 1是文本
            prediction_scores, seq_relationship_score = self.cls(
                sequence_output[1], pooled_output, task_idx=task_idx)
            # 测试时
            if next_sentence_label == None:
                return sequence_output,seq_relationship_score
            next_sentence_loss = self.crit_next_sent(
                    seq_relationship_score.view(-1, self.num_labels).float(), next_sentence_label.view(-1))
            masked_lm_loss = next_sentence_loss.new(1).fill_(0)
            
            return sequence_output,next_sentence_loss,seq_relationship_score
        elif self.loss_type == 'ctc_nsp_mlm':
            if masked_lm_labels == None or masked_pos.numel() == 0:
                # hack to avoid empty masked_pos during training for now
                masked_lm_loss = pooled_output.new(1).fill_(0)
                prediction_scores, seq_relationship_score = self.cls(
                    sequence_output[1], pooled_output, task_idx=task_idx)
                
            else:
                sequence_output_masked = gather_seq_out_by_pos(
                    sequence_output[1], masked_pos)
                prediction_scores_masked, seq_relationship_score = self.cls(
                    sequence_output_masked, pooled_output, task_idx=task_idx)
                if self.crit_mask_lm_smoothed:
                    masked_lm_loss = self.crit_mask_lm_smoothed(
                        F.log_softmax(prediction_scores_masked.float(), dim=-1), masked_lm_labels)
                else:
                    masked_lm_loss = self.crit_mask_lm(
                        prediction_scores_masked.transpose(1, 2).float(), masked_lm_labels)
                
                
                masked_lm_loss = loss_mask_and_normalize(
                    masked_lm_loss.float(), masked_weights, drop_worst_ratio)
                # 如果没有做mask，则将loss强制为0
                if masked_weights.sum() == 0:
                    masked_lm_loss = 0
            
            if next_sentence_label == None or masked_lm_labels == None:
                return sequence_output,seq_relationship_score
                
            # prediction_scores, seq_relationship_score = self.cls(
            #     sequence_output, pooled_output, task_idx=task_idx)
            next_sentence_loss = self.crit_next_sent(
                    seq_relationship_score.view(-1, self.num_labels).float(), next_sentence_label.view(-1))
            
            
            return sequence_output,next_sentence_loss,seq_relationship_score,masked_lm_loss



class BertHybridFusionNoShare(PreTrainedBertModel):
    def __init__(self, config, num_labels=2, enable_butd=False, len_vis_input=36,
                 visdial_v='1.0', loss_type='mlm', eval_disc=False, float_nsp_label=False,
                 neg_num=0, adaptive_weight=False, add_attn_fuse=False,
                 no_h0=False, add_val=False, no_vision=False, rank_loss='',**kwargs):
        super(BertHybridFusionNoShare, self).__init__(config)
        self.bert = BertModelHybridNoShare(config)
        self.cls = BertPreTrainingHeads(
            config, self.bert.embeddings.word_embeddings.weight,
            num_labels=num_labels)  # num_labels not applicable for VLP
        self.apply(self.init_bert_weights)
        self.crit_mask_lm = nn.CrossEntropyLoss(reduction='none')
        self.num_labels = num_labels
        self.len_vis_input = len_vis_input
        self.enable_butd = enable_butd
        self.eval_disc = eval_disc
        self.loss_type = loss_type
        self.visdial_v = visdial_v
        self.float_nsp_label = float_nsp_label
        self.add_attn_fuse = add_attn_fuse
        self.no_h0 = no_h0
        self.add_val = add_val
        self.no_vision = no_vision
        self.rank_loss = rank_loss
        self.dropout = nn.Dropout(config.hidden_dropout_prob)  # for self_attn

        if hasattr(config, 'label_smoothing') and config.label_smoothing:
            self.crit_mask_lm_smoothed = LabelSmoothingLoss(
                config.label_smoothing, config.vocab_size, ignore_index=0, reduction='none')
        else:
            self.crit_mask_lm_smoothed = None

        if 'nsp' in loss_type:
            if self.float_nsp_label:
                if self.rank_loss == 'softmax':
                    self.ce_loss_fct = nn.KLDivLoss(reduction='batchmean')
                else:
                    self.crit_next_sent = nn.CrossEntropyLoss(ignore_index=-1, reduction='none')
            else:
                if adaptive_weight and neg_num > 1:
                    weight = [2.0 / (1.0 + neg_num), 2 * neg_num / (1.0 + neg_num)]
                    print("Setting adaptive weights for neg/pos=%.2f/%.2f" % (weight[0], weight[1]))
                    weight = torch.tensor(weight, dtype=torch.float32).cuda()
                    self.crit_next_sent = nn.CrossEntropyLoss(ignore_index=-1, weight=weight)

                else:
                    self.crit_next_sent = nn.CrossEntropyLoss(ignore_index=-1)
        self.logsoftmax = nn.LogSoftmax(dim=-1)


    def forward(self, vis_feat,text_feat,position_ids, attention_mask_vis, attention_mask_text, attention_mask_VasQ, masked_lm_labels=None,
                next_sentence_label=None, masked_pos=None, masked_weights=None, task_idx=None,
                vis_masked_pos=[], mask_image_regions=False, drop_worst_ratio=0.2,len_vis_input = 36):

        
        def gather_seq_out_by_pos(seq, pos):
            return torch.gather(seq, 1, pos.unsqueeze(2).expand(-1, -1, seq.size(-1)))
   
        sequence_output, pooled_output = self.bert(
            vis_feat, text_feat, position_ids, 
            attention_mask_vis, attention_mask_text, attention_mask_VasQ,
            output_all_encoded_layers=False
        )
        
        # print(sequence_output.size())
        # vd-bert中bs为32，使用了drop worst，slr的bs只有2，所以不用
        def loss_mask_and_normalize(loss, mask, drop_worst_ratio):
            mask = mask.type_as(loss)
            loss = loss * mask

            return loss.sum() / mask.sum()

        # def loss_mask_and_normalize(loss, mask, drop_worst_ratio):
        #     mask = mask.type_as(loss)
        #     loss = loss * mask

        #     # Ruotian Luo's drop worst
        #     keep_loss, keep_ind = torch.topk(loss.sum(-1), int(loss.size(0) * (1 - drop_worst_ratio)), largest=False)

        #     # denominator = torch.sum(mask) + 1e-5
        #     # return (loss / denominator).sum()
        #     denominator = torch.sum(mask.sum(-1)[keep_ind]) + 1e-5
        #     return (keep_loss / denominator).sum()

        # masked lm
        if self.loss_type == 'ctc':
            return sequence_output
        elif self.loss_type == 'ctc_nsp':
            # sequence_output[0] 是视频 1是文本
            prediction_scores, seq_relationship_score = self.cls(
                sequence_output[1], pooled_output, task_idx=task_idx)
            # 测试时
            if next_sentence_label == None:
                return sequence_output,seq_relationship_score
            next_sentence_loss = self.crit_next_sent(
                    seq_relationship_score.view(-1, self.num_labels).float(), next_sentence_label.view(-1))
            masked_lm_loss = next_sentence_loss.new(1).fill_(0)
            
            return sequence_output,next_sentence_loss,seq_relationship_score
        elif self.loss_type == 'ctc_nsp_mlm':
            if masked_lm_labels == None or masked_pos.numel() == 0:
                # hack to avoid empty masked_pos during training for now
                masked_lm_loss = pooled_output.new(1).fill_(0)
                prediction_scores, seq_relationship_score = self.cls(
                    sequence_output[1], pooled_output, task_idx=task_idx)
                
            else:
                sequence_output_masked = gather_seq_out_by_pos(
                    sequence_output[1], masked_pos)
                prediction_scores_masked, seq_relationship_score = self.cls(
                    sequence_output_masked, pooled_output, task_idx=task_idx)
                if self.crit_mask_lm_smoothed:
                    masked_lm_loss = self.crit_mask_lm_smoothed(
                        F.log_softmax(prediction_scores_masked.float(), dim=-1), masked_lm_labels)
                else:
                    masked_lm_loss = self.crit_mask_lm(
                        prediction_scores_masked.transpose(1, 2).float(), masked_lm_labels)
                
                
                masked_lm_loss = loss_mask_and_normalize(
                    masked_lm_loss.float(), masked_weights, drop_worst_ratio)
                # 如果没有做mask，则将loss强制为0
                if masked_weights.sum() == 0:
                    masked_lm_loss = 0
            
            if next_sentence_label == None or masked_lm_labels == None:
                return sequence_output,seq_relationship_score
                
            # prediction_scores, seq_relationship_score = self.cls(
            #     sequence_output, pooled_output, task_idx=task_idx)
            next_sentence_loss = self.crit_next_sent(
                    seq_relationship_score.view(-1, self.num_labels).float(), next_sentence_label.view(-1))
            
            
            return sequence_output,next_sentence_loss,seq_relationship_score,masked_lm_loss