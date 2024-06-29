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
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(hidden_size,1)
        self.sigmoid = nn.Sigmoid()



    def forward(self, feature_combine):
        mask = self.conv(feature_combine)
        mask = self.relu(mask)
        mask = self.fc(mask.transpose(1,2))
        mask = self.sigmoid(mask).squeeze(-1)

        # mask_one = torch.ones_like(mask)
        feature_one = torch.einsum('ij,ijk->ijk',[mask,feature_combine.transpose(1,2)])
        feature_two = torch.einsum('ij,ijk->ijk',[(1-mask),feature_combine.transpose(1,2)])

        return feature_one.transpose(0,1),feature_two.transpose(0,1),mask
        
    
