import pdb
import copy
import torch
import collections
import torch.nn as nn
import torch.nn.functional as F

import collections

class FeatureDisentangle(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(FeatureDisentangle, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size


        self.conv = nn.Conv1d(input_size, self.hidden_size, kernel_size=5, stride=1, padding=2)
        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.relu = nn.ReLU(inplace=True)
        self.fc_1 = nn.Linear(hidden_size,hidden_size*2)
        self.fc_2 = nn.Linear(hidden_size*2,hidden_size)
        self.sigmoid = nn.Sigmoid()



    def forward(self, feature_combine):
        mask = self.conv(feature_combine)
        mask = self.pooling(mask)
        mask = mask.squeeze(-1)
        # mask = self.relu(mask)
        mask = self.fc_1(mask)
        mask = self.relu(mask)
        mask = self.fc_2(mask)
        mask = self.sigmoid(mask)

        # mask_one = torch.ones_like(mask)
        feature_one = torch.einsum('ik,ijk->ijk',[mask,feature_combine.transpose(1,2)])
        feature_two = torch.einsum('ik,ijk->ijk',[(1-mask),feature_combine.transpose(1,2)])

        return feature_one.transpose(0,1),feature_two.transpose(0,1),mask
        
    
