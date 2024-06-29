import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    """
    NLL loss with label smoothing.
    """

    def __init__(self):
        super(ContrastiveLoss, self).__init__()
        # self.kdloss = nn.KLDivLoss(reduction='batchmean')
        # self.T = T

    def forward(self, feature_ori, feature_pos, feature_neg, feature_neg2):
        pos = torch.cosine_similarity(feature_ori, feature_pos, dim=2) # [length,batch]
        neg = torch.cosine_similarity(feature_ori, feature_neg, dim=2) # [length,batch]
        neg2 = torch.cosine_similarity(feature_ori, feature_neg2, dim=2) # [length,batch]
        logit = torch.stack((pos, neg , neg2), 2) # [length,batch,3]
        softmax_logit = nn.functional.softmax(logit, 2) # [length,batch,3] 沿着最后一个维度做softmax
# softmax_logit[:,:,0] 表示exp(pos)/exp(pos)+exp(neg)+exp(neg2) 
        contras_loss = - torch.log(softmax_logit[:,:,0])
                    # contras_loss += torch.log(softmax_logit[:, 1]) # add contras_neg
        contras_loss = contras_loss.mean()
        return contras_loss

