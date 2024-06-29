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
        
        #  输入为B*C*Length
        self.instance_norm=nn.InstanceNorm1d(hidden_size)


        self.conv = nn.Conv1d(input_size, self.hidden_size, kernel_size=5, stride=1, padding=2)
        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.relu = nn.ReLU(inplace=True)
        self.fc_1 = nn.Linear(hidden_size,hidden_size*2)
        self.fc_2 = nn.Linear(hidden_size*2,hidden_size)
        self.sigmoid = nn.Sigmoid()



    def forward(self, feature_combine):
        # feature_combine b*hidden_size*length
        domain_invariant_feature=self.instance_norm(feature_combine)
        
        residual_feature=feature_combine-domain_invariant_feature;

        mask = self.conv(residual_feature)
        mask = self.pooling(mask)
        mask = mask.squeeze(-1)
        # mask = self.relu(mask)
        mask = self.fc_1(mask)
        mask = self.relu(mask)
        mask = self.fc_2(mask)
        mask = self.sigmoid(mask)

        # mask_one = torch.ones_like(mask)
        feature_relevant = torch.einsum('ik,ijk->ijk',[mask,residual_feature.transpose(1,2)])
        feature_irrelavant = torch.einsum('ik,ijk->ijk',[(1-mask),residual_feature.transpose(1,2)])
        
        
        feature_task = feature_relevant+domain_invariant_feature.transpose(1,2);
        feature_contaminated = feature_irrelavant+domain_invariant_feature.transpose(1,2);

        return feature_task.transpose(0,1),feature_contaminated.transpose(0,1),mask
        
    
