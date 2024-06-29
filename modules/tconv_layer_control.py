import pdb
import copy
import torch
import collections
import torch.nn as nn
import torch.nn.functional as F

import collections

class TemporalConv(nn.Module):
    def __init__(self, input_size, hidden_size, conv_type=4, use_bn=False, num_classes=-1):
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

        # self.modules = []      
        modules=[]    
        modules.append(
            nn.Conv1d(self.input_size, self.hidden_size, kernel_size=int(5), stride=1, padding=0)
        )
        modules.append(nn.BatchNorm1d(self.hidden_size))
        modules.append(nn.ReLU(inplace=True))
        self.K5_0 = nn.Sequential(*modules)

        modules=[] 
        modules.append(nn.MaxPool1d(kernel_size=int(2), ceil_mode=False))
        self.P2_0 = nn.Sequential(*modules)

        modules=[]    
        modules.append(
            nn.Conv1d(self.hidden_size, self.hidden_size, kernel_size=int(5), stride=1, padding=0)
        )
        modules.append(nn.BatchNorm1d(self.hidden_size))
        modules.append(nn.ReLU(inplace=True))
        self.K5_1 = nn.Sequential(*modules)

        modules=[] 
        modules.append(nn.MaxPool1d(kernel_size=int(2), ceil_mode=False))
        self.P2_1 = nn.Sequential(*modules)

        modules=[]    
        modules.append(
            nn.Conv1d(self.hidden_size, self.hidden_size, kernel_size=int(5), stride=1, padding=0)
        )
        modules.append(nn.BatchNorm1d(self.hidden_size))
        modules.append(nn.ReLU(inplace=True))
        self.K5_2 = nn.Sequential(*modules)

        modules=[] 
        modules.append(nn.MaxPool1d(kernel_size=int(2), ceil_mode=False))
        self.P2_2 = nn.Sequential(*modules)



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
    def forward(self, frame_feat, lgt, use_layers=None):
        if use_layers==None:
            use_layers=4
            # visual_feat = self.temporal_conv(frame_feat)
            # lgt = self.update_lgt(lgt)
            # logits = None if self.num_classes == -1 \
            #     else self.fc(visual_feat.transpose(1, 2)).transpose(1, 2)
            # return {
            #     "visual_feat": visual_feat.permute(2, 0, 1),
            #     "conv_logits": logits.permute(2, 0, 1),
            #     "feat_len": lgt.cpu(),
            # }
        
        visual_feat=frame_feat
        # visual_feat = self.temporal_conv(frame_feat)
        # for module_list in self.modules[0:use_layers]:
        #     visual_feat=module_list(visual_feat)
        if use_layers==4:
            visual_feat=self.K5_0(visual_feat)
            visual_feat=self.P2_0(visual_feat)
            visual_feat=self.K5_1(visual_feat)
            visual_feat=self.P2_1(visual_feat)
        elif use_layers==6:
            visual_feat=self.K5_0(visual_feat)
            visual_feat=self.P2_0(visual_feat)
            visual_feat=self.K5_1(visual_feat)
            visual_feat=self.P2_1(visual_feat)
            visual_feat=self.K5_2(visual_feat)
            visual_feat=self.P2_2(visual_feat)


        lgt = self.update_lgt(lgt,use_layers)
        logits = None if self.num_classes == -1 \
            else self.fc(visual_feat.transpose(1, 2)).transpose(1, 2)
        return {
            "visual_feat": visual_feat.permute(2, 0, 1),
            "conv_logits": logits.permute(2, 0, 1),
            "feat_len": lgt.cpu(),
        }
    
    
