import torch
import torch.nn.functional as F
import numpy as np


batch_size = 4
length = 100
hidden_size = 768
# 手语feature batch_size 4 length 100 hidden_size 768
vis_feats_origin = torch.randn(batch_size, length, hidden_size)


# # mask，随机选择15%的位置,不能保证每一个clip都是选了15%
# mask = np.random.choice([0,1],[batch_size, length, hidden_size],p=[0.15,0.85])
# mask = torch.from_numpy(mask)

# vis_feats_masked = vis_feats_origin * mask

# print('----')

# 保证每一个clip都选择固定的比例
mask = torch.ones_like(vis_feats_origin)
for i in range(0,batch_size):
    for j in range(0,length):
        mask_one = np.random.choice(768,[120],replace = False)
        mask_one_tensor = torch.from_numpy(mask_one)
        mask[i,j,mask_one_tensor] = 0
vis_feats_masked = vis_feats_origin * mask

