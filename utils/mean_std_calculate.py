import cv2
import pdb
import PIL
import copy
import scipy.misc
import glob
import torch
import random
import numbers
import numpy as np
import os
# from torchvision import transforms


def read_video(img_folder):
    img_list = sorted(glob.glob(img_folder))
    return [cv2.cvtColor(cv2.imread(i), cv2.COLOR_BGR2RGB) for i in img_list]


train_txt=open('/home/gaoliqing/shipeng/code/VAC_CSLR_TE/DatasetFile/Split_insert_com_T_E_multiword_new/200TE_train.txt')
template_list=[]
entity_list=[]
channel_mean = torch.zeros(3)
channel_std = torch.zeros(3)
for i in train_txt.readlines():
    template_list.append(i.strip().split('Q')[0])
    entity_list.append(i.strip().split('Q')[1])
num=0
for i in template_list:
    img_folder=os.path.join(i,"*.jpg")
    imagelist=read_video(img_folder)
    video = np.array(imagelist)
    video = torch.from_numpy(video.transpose((0, 3, 1, 2))).float()
    video = video/255
    batch,channel,width,height=video.size()
    video=video.view(batch,channel,-1)
    batch_channel_mean=torch.mean(video,2)
    channel_mean+=torch.mean(batch_channel_mean,0)
    batch_channel_std=torch.std(video,2)
    channel_std+=torch.mean(batch_channel_std,0)
    num+=1
    print(num)
for i in entity_list:
    img_folder=os.path.join(i,"*.jpg")
    imagelist=read_video(img_folder)
    video = np.array(imagelist)
    video = torch.from_numpy(video.transpose((0, 3, 1, 2))).float()
    video = video/255
    batch,channel,width,height=video.size()
    video=video.view(batch,channel,-1)
    batch_channel_mean=torch.mean(video,2)
    channel_mean+=torch.mean(batch_channel_mean,0)
    batch_channel_std=torch.std(video,2)
    channel_std+=torch.mean(batch_channel_std,0)

channel_mean=channel_mean/(len(template_list)+len(entity_list))
channel_std=channel_std/(len(template_list)+len(entity_list))
print('channel_mean',channel_mean)
print('channel_std',channel_std)
    

    
    

