import pdb
import copy
import utils
import torch
import types
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from modules.tconv import TemporalConv, TemporalConv2SignDict
from modules import BiLSTMLayer
from modules.criterions import SeqKD

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        # self.fc = nn.Linear(1024,768)

    def forward(self, x):
        # x = self.fc(x)
        return x

class VACModel(nn.Module):
    def __init__(self, num_classes,c2d_type, conv_type, use_bn=False, tm_type='BiLSTM',
                 hidden_size= 1024, gloss_dict=None):
        super(VACModel, self).__init__()
        self.num_classes = num_classes
        self.conv2d = getattr(models, c2d_type)(pretrained=True)
        self.conv2d.fc = Identity()
        self.conv1d = TemporalConv(input_size=512,
                                   hidden_size=hidden_size,
                                   conv_type=conv_type,
                                   use_bn=use_bn,
                                   num_classes=num_classes)
        self.temporal_model = BiLSTMLayer(rnn_type='LSTM', input_size=hidden_size, hidden_size=hidden_size,
                                              num_layers=2, bidirectional=True)
        self.gloss_dict=gloss_dict
        self.gloss_dict_word2index=dict((v, k) for k, v in gloss_dict.items())
        self.classifier_sign = nn.Linear(hidden_size, self.num_classes)
        self.register_backward_hook(self.backward_hook)

    def backward_hook(self, module, grad_input, grad_output):
        for g in grad_input:
            g[g != g] = 0

    def masked_bn(self, inputs, len_x):
        def pad(tensor, length):
            return torch.cat([tensor, tensor.new(length - tensor.size(0), *tensor.size()[1:]).zero_()])

        x = torch.cat([inputs[max(len_x) * idx:max(len_x) * idx + lgt] for idx, lgt in enumerate(len_x)])
        x = self.conv2d(x)
        x = torch.cat([pad(x[sum(len_x[:idx]):sum(len_x[:idx + 1])], max(len_x))
                       for idx, lgt in enumerate(len_x)])

        # x = torch.cat([inputs[len_x[0] * idx:len_x[0] * idx + lgt] for idx, lgt in enumerate(len_x)])
        # x = self.conv2d(x)
        # x = torch.cat([pad(x[sum(len_x[:idx]):sum(len_x[:idx + 1])], len_x[0])
        #                for idx, lgt in enumerate(len_x)])
        return x

    def forward(self, x, len_x_all):
            # videos
        batch, temp, channel, height, width = x.shape
        inputs = x.reshape(batch * temp, channel, height, width)
        framewise_all = self.masked_bn(inputs, len_x_all)
        framewise_all = framewise_all.reshape(batch, temp, -1)


        framewise = framewise_all
        len_x = len_x_all
        framewise = framewise.transpose(1, 2)
        # bs*dim*length
        conv1d_outputs = self.conv1d(framewise, len_x)
        conv_output = conv1d_outputs['visual_feat']
        lgt = conv1d_outputs['feat_len']

        tm_outputs = self.temporal_model(conv_output, lgt)
        outputs_sign = self.classifier_sign(tm_outputs['predictions'])

        return {
            "feat_len": lgt,
            "conv_logits" : conv1d_outputs['conv_logits'],
            "sign_feat": tm_outputs['predictions'],
            "sequence_logits_sign" : outputs_sign,
        }
    def forward_framewise(self, x, len_x_all):
            # videos
        batch, temp, channel, height, width = x.shape
        inputs = x.reshape(batch * temp, channel, height, width)
        framewise_all = self.masked_bn(inputs, len_x_all)
        framewise_all = framewise_all.reshape(batch, temp, -1)


        framewise = framewise_all
        len_x = len_x_all
        framewise = framewise.transpose(1, 2)
        # bs*dim*length
        conv1d_outputs = self.conv1d(framewise, len_x)
        conv_output = conv1d_outputs['visual_feat']
        lgt = conv1d_outputs['feat_len']

        tm_outputs = self.temporal_model(conv_output, lgt)
        outputs_sign = self.classifier_sign(tm_outputs['predictions'])

        return {
            "framewise" : framewise,
            "feat_len": lgt,
            "conv_logits" : conv1d_outputs['conv_logits'],
            "sign_feat": tm_outputs['predictions'],
            "sequence_logits_sign" : outputs_sign,
        }
    
    def forward_framewiseOnly(self, x, len_x_all):
            # videos
        batch, temp, channel, height, width = x.shape
        inputs = x.reshape(batch * temp, channel, height, width)
        framewise_all = self.masked_bn(inputs, len_x_all)
        framewise_all = framewise_all.reshape(batch, temp, -1)


        framewise = framewise_all
        len_x = len_x_all
        framewise = framewise.transpose(1, 2)
        # bs*dim*length

        return {
            "framewise" : framewise,

        }

    def forward_noRNN(self, x, len_x_all):
            # videos
        batch, temp, channel, height, width = x.shape
        inputs = x.reshape(batch * temp, channel, height, width)
        framewise_all = self.masked_bn(inputs, len_x_all)
        framewise_all = framewise_all.reshape(batch, temp, -1)


        framewise = framewise_all
        len_x = len_x_all
        framewise = framewise.transpose(1, 2)
        # bs*dim*length
        conv1d_outputs = self.conv1d(framewise, len_x)
        conv_output = conv1d_outputs['visual_feat']
        lgt = conv1d_outputs['feat_len']

        tm_outputs = self.temporal_model(conv_output, lgt)
        outputs_sign = self.classifier_sign(tm_outputs['predictions'])

        return {
            "feat_len": lgt,
            "conv_logits" : conv1d_outputs['conv_logits'],
            "sign_feat": conv_output,
            "sequence_logits_sign" : outputs_sign,
        }
class VACModel2SignDict(nn.Module):
    def __init__(self, num_classes_fine,num_classes_coarse,c2d_type, conv_type, use_bn=False, tm_type='BiLSTM',
                 hidden_size= 1024, gloss_dict_fine=None,gloss_dict_coarse=None):
        super(VACModel2SignDict, self).__init__()
        self.num_classes_fine = num_classes_fine
        self.num_classes_coarse = num_classes_coarse
        self.conv2d = getattr(models, c2d_type)(pretrained=True)
        self.conv2d.fc = Identity()
        self.conv1d = TemporalConv2SignDict(input_size=512,
                                   hidden_size=hidden_size,
                                   conv_type=conv_type,
                                   use_bn=use_bn,
                                   num_classes_fine=num_classes_fine,
                                   num_classes_coarse=num_classes_coarse)
        self.temporal_model = BiLSTMLayer(rnn_type='LSTM', input_size=hidden_size, hidden_size=hidden_size,
                                              num_layers=2, bidirectional=True)
        # self.gloss_dict_fine=gloss_dict_fine
        # self.gloss_dict_word2index=dict((v, k) for k, v in gloss_dict.items())
        self.classifier_sign_fine = nn.Linear(hidden_size, self.num_classes_fine)
        self.classifier_sign_coarse= nn.Linear(hidden_size, self.num_classes_coarse)
        self.register_backward_hook(self.backward_hook)

    def backward_hook(self, module, grad_input, grad_output):
        for g in grad_input:
            g[g != g] = 0

    def masked_bn(self, inputs, len_x):
        def pad(tensor, length):
            return torch.cat([tensor, tensor.new(length - tensor.size(0), *tensor.size()[1:]).zero_()])

        x = torch.cat([inputs[max(len_x) * idx:max(len_x) * idx + lgt] for idx, lgt in enumerate(len_x)])
        x = self.conv2d(x)
        x = torch.cat([pad(x[sum(len_x[:idx]):sum(len_x[:idx + 1])], max(len_x))
                       for idx, lgt in enumerate(len_x)])

        # x = torch.cat([inputs[len_x[0] * idx:len_x[0] * idx + lgt] for idx, lgt in enumerate(len_x)])
        # x = self.conv2d(x)
        # x = torch.cat([pad(x[sum(len_x[:idx]):sum(len_x[:idx + 1])], len_x[0])
        #                for idx, lgt in enumerate(len_x)])
        return x

    def forward(self, x, len_x_all):
            # videos
        batch, temp, channel, height, width = x.shape
        inputs = x.reshape(batch * temp, channel, height, width)
        framewise_all = self.masked_bn(inputs, len_x_all)
        framewise_all = framewise_all.reshape(batch, temp, -1)


        framewise = framewise_all
        len_x = len_x_all
        framewise = framewise.transpose(1, 2)
        # bs*dim*length
        conv1d_outputs = self.conv1d(framewise, len_x)
        conv_output = conv1d_outputs['visual_feat']
        lgt = conv1d_outputs['feat_len']

        tm_outputs = self.temporal_model(conv_output, lgt)
        outputs_sign_fine = self.classifier_sign_fine(tm_outputs['predictions'])
        outputs_sign_coarse = self.classifier_sign_coarse(tm_outputs['predictions'])

        return {
            "feat_len": lgt,
            "conv_logits_fine" : conv1d_outputs['conv_logits_fine'],
            "conv_logits_coarse": conv1d_outputs['conv_logits_coarse'],
            "sign_feat": tm_outputs['predictions'],
            "sequence_logits_sign_fine" : outputs_sign_fine,
            "sequence_logits_sign_coarse" : outputs_sign_coarse,
        }



class VACModelNoCNN(nn.Module):
    def __init__(self, num_classes,c2d_type, conv_type, use_bn=False, tm_type='BiLSTM',
                 hidden_size= 1024, gloss_dict=None):
        super(VACModelNoCNN, self).__init__()
        self.num_classes = num_classes
        # self.conv2d = getattr(models, c2d_type)(pretrained=True)
        # self.conv2d.fc = Identity()
        self.conv1d = TemporalConv(input_size=512,
                                   hidden_size=hidden_size,
                                   conv_type=conv_type,
                                   use_bn=use_bn,
                                   num_classes=num_classes)
        self.temporal_model = BiLSTMLayer(rnn_type='LSTM', input_size=hidden_size, hidden_size=hidden_size,
                                              num_layers=2, bidirectional=True)
        self.gloss_dict=gloss_dict
        self.gloss_dict_word2index=dict((v, k) for k, v in gloss_dict.items())
        self.classifier_sign = nn.Linear(hidden_size, self.num_classes)
        self.register_backward_hook(self.backward_hook)

    def backward_hook(self, module, grad_input, grad_output):
        for g in grad_input:
            g[g != g] = 0

    def masked_bn(self, inputs, len_x):
        def pad(tensor, length):
            return torch.cat([tensor, tensor.new(length - tensor.size(0), *tensor.size()[1:]).zero_()])

        x = torch.cat([inputs[max(len_x) * idx:max(len_x) * idx + lgt] for idx, lgt in enumerate(len_x)])
        x = self.conv2d(x)
        x = torch.cat([pad(x[sum(len_x[:idx]):sum(len_x[:idx + 1])], max(len_x))
                       for idx, lgt in enumerate(len_x)])

        # x = torch.cat([inputs[len_x[0] * idx:len_x[0] * idx + lgt] for idx, lgt in enumerate(len_x)])
        # x = self.conv2d(x)
        # x = torch.cat([pad(x[sum(len_x[:idx]):sum(len_x[:idx + 1])], len_x[0])
        #                for idx, lgt in enumerate(len_x)])
        return x

    def forward(self, framewise, len_x):
            # videos
        # batch, temp, channel, height, width = x.shape
        # inputs = x.reshape(batch * temp, channel, height, width)
        # framewise_all = self.masked_bn(inputs, len_x_all)
        # framewise_all = framewise_all.reshape(batch, temp, -1)


        # framewise = framewise_all
        # len_x = len_x_all
        # framewise = framewise.transpose(1, 2)
        # bs*dim*length
        conv1d_outputs = self.conv1d(framewise, len_x)
        conv_output = conv1d_outputs['visual_feat']
        lgt = conv1d_outputs['feat_len']

        tm_outputs = self.temporal_model(conv_output, lgt)
        outputs_sign = self.classifier_sign(tm_outputs['predictions'])

        return {
            "feat_len": lgt,
            "conv_logits" : conv1d_outputs['conv_logits'],
            "sign_feat": tm_outputs['predictions'],
            "sequence_logits_sign" : outputs_sign,
        }