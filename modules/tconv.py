import pdb
import copy
import torch
import collections
import torch.nn as nn
import torch.nn.functional as F

import collections

class TemporalConv(nn.Module):
    def __init__(self, input_size, hidden_size, conv_type=2, use_bn=False, num_classes=-1):
        super(TemporalConv, self).__init__()
        self.use_bn = use_bn
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.conv_type = conv_type

        if self.conv_type == 0:
            self.kernel_size = ['K3']
        elif self.conv_type == 1:
            self.kernel_size = ['K5', "P2"]
        elif self.conv_type == 2:
            self.kernel_size = ['K5', "P2", 'K5', "P2"]
        elif self.conv_type == 3:
            self.kernel_size = ['K6', "P2", 'K6', "P2"]
        elif self.conv_type == 4:
            self.kernel_size = ['K5', "P2", 'K5', "P2",'K5', "P2"]
        elif self.conv_type == 5:
            self.kernel_size = ['K5', "P2", 'K5']

        self.modules = []
        for layer_idx, ks in enumerate(self.kernel_size):
            input_sz = self.input_size if layer_idx == 0 else self.hidden_size
            if ks[0] == 'P':
                self.modules.append(nn.MaxPool1d(kernel_size=int(ks[1]), ceil_mode=False))
            elif ks[0] == 'K':
                self.modules.append(
                    nn.Conv1d(input_sz, self.hidden_size, kernel_size=int(ks[1]), stride=1, padding=0)
                )
                self.modules.append(nn.BatchNorm1d(self.hidden_size))
                self.modules.append(nn.ReLU(inplace=True))
        self.temporal_conv = nn.Sequential(*self.modules)

        if self.num_classes != -1:
            self.fc = nn.Linear(self.hidden_size, self.num_classes)

    def update_lgt(self, lgt,use_layers=None):
        if use_layers==None:
            feat_len = copy.deepcopy(lgt)
            for ks in self.kernel_size:
                if ks[0] == 'P':
                    feat_len //= 2
                else:
                    feat_len -= int(ks[1]) - 1
            return feat_len
        else:
            feat_len = copy.deepcopy(lgt)
            for ks in self.kernel_size[0:use_layers]:
                if ks[0] == 'P':
                    feat_len //= 2
                else:
                    feat_len -= int(ks[1]) - 1
            return feat_len

    # use_layers: 指定使用几层卷积层
    def forward(self, frame_feat, lgt):
        visual_feat = self.temporal_conv(frame_feat)
        lgt = self.update_lgt(lgt)
        logits = None if self.num_classes == -1 \
            else self.fc(visual_feat.transpose(1, 2)).transpose(1, 2)
        return {
            "visual_feat": visual_feat.permute(2, 0, 1),
            "conv_logits": logits.permute(2, 0, 1),
            "feat_len": lgt.cpu(),
        }

class TemporalConv2SignDict(nn.Module):
    def __init__(self, input_size, hidden_size, conv_type=2, use_bn=False, num_classes_fine = -1, num_classes_coarse = -1):
        super(TemporalConv2SignDict, self).__init__()
        self.use_bn = use_bn
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes_fine = num_classes_fine
        self.num_classes_coarse = num_classes_coarse
        self.conv_type = conv_type

        if self.conv_type == 0:
            self.kernel_size = ['K3']
        elif self.conv_type == 1:
            self.kernel_size = ['K5', "P2"]
        elif self.conv_type == 2:
            self.kernel_size = ['K5', "P2", 'K5', "P2"]
        elif self.conv_type == 3:
            self.kernel_size = ['K6', "P2", 'K6', "P2"]
        elif self.conv_type == 4:
            self.kernel_size = ['K5', "P2", 'K5', "P2",'K5', "P2"]
        elif self.conv_type == 5:
            self.kernel_size = ['K5', "P2", 'K5']

        self.modules = []
        for layer_idx, ks in enumerate(self.kernel_size):
            input_sz = self.input_size if layer_idx == 0 else self.hidden_size
            if ks[0] == 'P':
                self.modules.append(nn.MaxPool1d(kernel_size=int(ks[1]), ceil_mode=False))
            elif ks[0] == 'K':
                self.modules.append(
                    nn.Conv1d(input_sz, self.hidden_size, kernel_size=int(ks[1]), stride=1, padding=0)
                )
                self.modules.append(nn.BatchNorm1d(self.hidden_size))
                self.modules.append(nn.ReLU(inplace=True))
        self.temporal_conv = nn.Sequential(*self.modules)

        if self.num_classes_fine != -1:
            self.fc_fine = nn.Linear(self.hidden_size, self.num_classes_fine)
        if self.num_classes_coarse != -1:
            self.fc_coarse = nn.Linear(self.hidden_size, self.num_classes_coarse)

    def update_lgt(self, lgt,use_layers=None):
        if use_layers==None:
            feat_len = copy.deepcopy(lgt)
            for ks in self.kernel_size:
                if ks[0] == 'P':
                    feat_len //= 2
                else:
                    feat_len -= int(ks[1]) - 1
            return feat_len
        else:
            feat_len = copy.deepcopy(lgt)
            for ks in self.kernel_size[0:use_layers]:
                if ks[0] == 'P':
                    feat_len //= 2
                else:
                    feat_len -= int(ks[1]) - 1
            return feat_len

    # def forward(self, frame_feat, lgt):
    #     visual_feat = self.temporal_conv(frame_feat)
    #     lgt = self.update_lgt(lgt)
    #     logits = None if self.num_classes == -1 \
    #         else self.fc(visual_feat.transpose(1, 2)).transpose(1, 2)
    #     return {
    #         "visual_feat": visual_feat.permute(2, 0, 1),
    #         "conv_logits": logits.permute(2, 0, 1),
    #         "feat_len": lgt.cpu(),
    #     }

    # use_layers: 指定使用几层卷积层
    def forward(self, frame_feat, lgt):
        visual_feat = self.temporal_conv(frame_feat)
        lgt = self.update_lgt(lgt)
        logits_fine = None if self.num_classes_fine == -1 \
            else self.fc_fine(visual_feat.transpose(1, 2)).transpose(1, 2)
        logits_coarse = None if self.num_classes_coarse == -1 \
            else self.fc_coarse(visual_feat.transpose(1, 2)).transpose(1, 2)
        return {
            "visual_feat": visual_feat.permute(2, 0, 1),
            "conv_logits_fine": logits_fine.permute(2, 0, 1),
            "conv_logits_coarse": logits_coarse.permute(2, 0, 1),
            "feat_len": lgt.cpu(),
        }
        

class TemporalConvText(nn.Module):
    def __init__(self, input_size, hidden_size, conv_type=2, use_bn=False, num_classes=-1):
        super(TemporalConvText, self).__init__()
        self.use_bn = use_bn
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.conv_type = conv_type

        if self.conv_type == 0:
            self.kernel_size = ['K3']
        elif self.conv_type == 1:
            self.kernel_size = ['K5', "P2"]
        elif self.conv_type == 2:
            self.kernel_size = ['K5', "P2", 'K5', "P2"]
        elif self.conv_type == 3:
            self.kernel_size = ['K6', "P2", 'K6', "P2"]
        elif self.conv_type == 4:
            self.kernel_size = ['K5', "P2", 'K5', "P2",'K5', "P2"]
        elif self.conv_type == 5:
            self.kernel_size = ['K5', "P2", 'K5']

        self.modules = []
        for layer_idx, ks in enumerate(self.kernel_size):
            input_sz = self.input_size if layer_idx == 0 else self.hidden_size
            if ks[0] == 'P':
                self.modules.append(nn.MaxPool1d(kernel_size=int(ks[1]), ceil_mode=False))
            elif ks[0] == 'K':
                self.modules.append(
                    nn.Conv1d(input_sz, self.hidden_size, kernel_size=int(ks[1]), stride=1, padding=0)
                )
                # self.modules.append(nn.BatchNorm1d(self.hidden_size))
                self.modules.append(nn.ReLU(inplace=True))
        self.temporal_conv = nn.Sequential(*self.modules)

        if self.num_classes != -1:
            self.fc = nn.Linear(self.hidden_size, self.num_classes)

    def update_lgt(self, lgt,use_layers=None):
        if use_layers==None:
            feat_len = copy.deepcopy(lgt)
            for ks in self.kernel_size:
                if ks[0] == 'P':
                    feat_len //= 2
                else:
                    feat_len -= int(ks[1]) - 1
            return feat_len
        else:
            feat_len = copy.deepcopy(lgt)
            for ks in self.kernel_size[0:use_layers]:
                if ks[0] == 'P':
                    feat_len //= 2
                else:
                    feat_len -= int(ks[1]) - 1
            return feat_len

    # use_layers: 指定使用几层卷积层
    def forward(self, frame_feat, lgt):
        visual_feat = self.temporal_conv(frame_feat)
        lgt = self.update_lgt(lgt)
        logits = None if self.num_classes == -1 \
            else self.fc(visual_feat.transpose(1, 2)).transpose(1, 2)
        return {
            "text_feat": visual_feat,
            "feat_len": lgt.cpu(),
        }
    
