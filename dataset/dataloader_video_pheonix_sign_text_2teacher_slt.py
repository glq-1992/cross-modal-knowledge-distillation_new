import os
import cv2
import sys
import pdb
import six
import glob
import time
import torch
import random
import pandas
import warnings
import copy



warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
# import pyarrow as pa
from PIL import Image
import torch.utils.data as data
import matplotlib.pyplot as plt
from utils import video_augmentation_TJU_insert as video_augmentation
from torch.utils.data.sampler import Sampler
from pytorch_pretrained_bert.tokenization import BertTokenizer, WhitespaceTokenizer

sys.path.append("..")

PAD_TOKEN = "<pad>"
BOS_TOKEN = "<s>"
EOS_TOKEN = "</s>"


class BaseFeeder(data.Dataset):
    def __init__(self, prefix, gloss_dict, text_dict,tokenizer,drop_ratio=1, num_gloss=-1, mode="train", transform_mode=True,
                 datatype="lmdb",frame_interval=1,meaningless_frame_begin=10,meaningless_frame_end=5,meaningless_frame_begin_E=10):
        self.mode = mode
        self.ng = num_gloss
        self.prefix = prefix
        self.dict = gloss_dict
        self.text_dict = text_dict
        self.itos = dict((v, k) for k, v in text_dict.items())
        self.tokenizer = tokenizer

        self.data_type = datatype
        self.dataset_root = '/disk1/dataset/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/fullFrame-256x256px/'
        # self.dataset_root = '/hd5/dataset/CSL-TJU_third_person_image_112'

        self.feat_prefix = f"{prefix}/features/fullFrame-256x256px/{mode}"
        self.transform_mode = "train" if transform_mode else "test"
        data_txt = f"/home/gaoliqing/shipeng/code/slrBert/DatasetFile/dialogue_pheonix/{mode}.txt"

        self.inputs_list={}

        self.frame_interval=frame_interval
        self.meaningless_frame_begin=meaningless_frame_begin
        self.meaningless_frame_end=meaningless_frame_end
        self.meaningless_frame_begin_E=meaningless_frame_begin_E

        with open(data_txt) as f:
            for i,line in enumerate(f.readlines()):
                line = line.replace('\n', '')
                folderAndTextLabel, text, label = line.split('%')
                folder = folderAndTextLabel.split(' ')[0]
                textLabel = ' '.join(folderAndTextLabel.split(' ')[1:])
                # folderAndLabel, text = line.split('%')
                # folder = folderAndLabel.split(' ')[0]
                # label = ' '.join(folderAndLabel.split(' ')[1:])
                self.inputs_list[i] = dict(folder=folder,label=label,textLabel = textLabel, text=text,next_sentence_label=1,original_info=folder)
        # if mode == 'train':
        #     pos_len = len(self.inputs_list)
        #     i = pos_len
        #     for j in range(0, pos_len):
        #         text_neg_j = random.choice(list(range(0, j)) + list(range(j+1, pos_len))) 
        #         text_neg = self.inputs_list[text_neg_j]['text']
        #         pos_dict = copy.deepcopy(self.inputs_list[j])
        #         while text_neg == pos_dict['text']:
        #             text_neg_j = random.choice(list(range(0, j)) + list(range(j+1, pos_len))) 
        #             text_neg = self.inputs_list[text_neg_j]['text']
        #             pos_dict = copy.deepcopy(self.inputs_list[j])
                
        #         # pos_dict = copy.deepcopy(self.inputs_list[j])
        #         pos_dict['text'] = text_neg
        #         # 负样本，对话上下文和手语不对应
        #         pos_dict['next_sentence_label'] =0
        #         self.inputs_list[i] = pos_dict
        #         i = i + 1

        self.data_aug = self.transform()

    def __getitem__(self, idx):
        input_data, label, text_label_list, text,next_sentence_label, fi = self.read_video(idx)
        input_data, label = self.normalize(Rvideo=input_data, label=label)
        # input_data, label = self.normalize(input_data, label, fi['fileid'])
        return input_data, label, text_label_list, text,next_sentence_label,self.inputs_list[idx]['original_info'],'test'


    def read_video(self, index):
        fi = self.inputs_list[index]
        # img_folder = fi['folder']
        img_folder =  self.dataset_root + fi['folder'] + '/*.png'
        img_list_all = sorted(glob.glob(img_folder))

        img_list = img_list_all[::self.frame_interval]

        label_list = []
        for phase in fi['label'].split(" "):
            if phase == '':
                continue
            if phase in self.dict.keys():
                label_list.append(self.dict[phase])

        text_label_list = []
        text_label_list.append(self.text_dict["<s>"])
        for phase in fi['textLabel'].split(" "):
            if phase == '':
                continue
            if phase in self.text_dict.keys():
                text_label_list.append(self.text_dict[phase])
        text_label_list.append(self.text_dict["</s>"])
        

        text_list = self.tokenizer.tokenize(fi['text'])
        # text_list = ['[CLS]'] +['[SEP]'] + text_list + ['[SEP]'] 
        text_list = ['[CLS]'] + text_list + ['[SEP]'] 

        input_ids = self.tokenizer.convert_tokens_to_ids(text_list)
      
        return [cv2.cvtColor(cv2.imread(i), cv2.COLOR_BGR2RGB) for i in img_list], label_list, text_label_list, input_ids,fi["next_sentence_label"], fi


    def read_features(self, index):
        # load file info
        fi = self.inputs_list[index]
        data = np.load(f"./features/{self.mode}/{fi['fileid']}_features.npy", allow_pickle=True).item()
        return data['features'], data['label']

    def normalize(self, Rvideo=None, label=None, file_id=None):
        Rvideo, label = self.data_aug(Rvideo, label, file_id)
        Rvideo = Rvideo.float() / 127.5 - 1
        # Rvideo = Rvideo.float() / 255

        return Rvideo, label



    # def transform(self):
    #     if self.transform_mode == "train":
    #         print("Apply training transform.")
    #         return video_augmentation.Compose([
    #             # video_augmentation.CenterCrop(224),
    #             # video_augmentation.WERAugment('/lustre/wangtao/current_exp/exp/baseline/boundary.npy'),
    #             # video_augmentation.RandomCrop(224),
    #             video_augmentation.RandomHorizontalFlip(0.5),
    #             video_augmentation.ToTensor(),
    #             # video_augmentation.TemporalRescale(0.2),
    #             # video_augmentation.Resize(0.5),
    #         ])
    #     else:
    #         print("Apply testing transform.")
    #         return video_augmentation.Compose([
    #             # video_augmentation.CenterCrop(224),
    #             # video_augmentation.Resize(0.5),
    #             video_augmentation.ToTensor(),
    #         ])
    # 新版
    def transform(self):
        if self.transform_mode == "train":
            print("Apply training transform.")
            return video_augmentation.Compose([
                # video_augmentation.CenterCrop(224),
                # video_augmentation.WERAugment('/lustre/wangtao/current_exp/exp/baseline/boundary.npy'),
                video_augmentation.RandomCrop(224),
                # video_augmentation.ResizeShape((224,224)),
                video_augmentation.RandomHorizontalFlip(0.5),
                # video_augmentation.Resize(1),
                video_augmentation.ToTensor(),
                video_augmentation.TemporalRescale(0.2),
                # video_augmentation.Resize(0.5),
            ])
        else:
            print("Apply testing transform.")
            return video_augmentation.Compose([
                # video_augmentation.ResizeShape((224,224)),
                video_augmentation.CenterCrop(224),
                # video_augmentation.Resize(0.5),
                video_augmentation.ToTensor(),
            ])
    # def byte_to_img(self, byteflow):
    #     unpacked = pa.deserialize(byteflow)
    #     imgbuf = unpacked[0]
    #     buf = six.BytesIO()
    #     buf.write(imgbuf)
    #     buf.seek(0)
    #     img = Image.open(buf).convert('RGB')
    #     return img

    @staticmethod
    def collate_fn(batch):
        
        batch = [item for item in sorted(batch, key=lambda x: len(x[0]), reverse=True)]
        video, label, text_label ,text , next_sentence_label,info,_ = list(zip(*batch))
        if len(video[0].shape) > 3:
            max_len = len(video[0])
            video_length = torch.LongTensor([np.ceil(len(vid) / 4.0) * 4 + 12 for vid in video])
            left_pad = 6
            right_pad = int(np.ceil(max_len / 4.0)) * 4 - max_len + 6
            max_len = max_len + left_pad + right_pad
            padded_video = [torch.cat(
                (
                    vid[0][None].expand(left_pad, -1, -1, -1),
                    vid,
                    vid[-1][None].expand(max_len - len(vid) - left_pad, -1, -1, -1),
                )
                , dim=0)
                for vid in video]
            padded_video = torch.stack(padded_video)
        else:
            max_len = len(video[0])
            video_length = torch.LongTensor([len(vid) for vid in video])
            padded_video = [torch.cat(
                (
                    vid,
                    vid[-1][None].expand(max_len - len(vid), -1),
                )
                , dim=0)
                for vid in video]
            padded_video = torch.stack(padded_video).permute(0, 2, 1)

        text_length = torch.LongTensor([len(t) for t in text])
        label_length = torch.LongTensor([len(lab) for lab in label])
        text_label_length = torch.LongTensor([len(t) - 1  for t in text_label]) # # 训练时decoder的输入不包括eos
        # if max(label_length) == 0:
        #     return padded_video, video_length, [], [], info
        
        padded_label = []
        padded_label_teacher = []
        for i in range(0,len(label)):
            padded_label.extend(label[i])
            padded_label_teacher.append(label[i] + [0] * (max(label_length) - label_length[i]))
        padded_label = torch.LongTensor(padded_label)
        padded_label_teacher = torch.LongTensor(padded_label_teacher)

        

        # padded_text = torch.zeros
        
        max_length_text = max(text_length)
        padded_text = []
        for i in range(0,len(text)):
            # [PAD]对应0
            padded_text.append( text[i] + [0] * (max_length_text - text_length[i]))
        padded_text = torch.LongTensor(padded_text)

        # 训练时decoder的输入不包括eos
        max_text_label_length = max(text_label_length)
        padded_text_label = []
        for i in range(0,len(text_label)):
            # [PAD]对应0
            padded_text_label.append( text_label[i][:-1] + [0] * (max_text_label_length - text_label_length[i]))
        padded_text_label = torch.LongTensor(padded_text_label)

        # 计算loss时不包括sos
        # max_text_label_length = max(text_label_length)
        padded_text_label_target = []
        for i in range(0,len(text_label)):
            # [PAD]对应0
            padded_text_label_target.append( text_label[i][1:] + [0] * (max_text_label_length - text_label_length[i]))
        padded_text_label_target = torch.LongTensor(padded_text_label_target)

        padded_next_sentence_label = []
        for i in next_sentence_label:
            padded_next_sentence_label.append(i)
        padded_next_sentence_label = torch.LongTensor(padded_next_sentence_label)
        return padded_video, video_length, padded_label, label_length, padded_label_teacher,padded_text_label,text_label_length,padded_text_label_target,padded_text,text_length,padded_next_sentence_label, info

    def __len__(self):
        return len(self.inputs_list)

    def record_time(self):
        self.cur_time = time.time()
        return self.cur_time

    def split_time(self):
        split_time = time.time() - self.cur_time
        self.record_time()
        return split_time

    def array_to_sentence(self, array: np.array, cut_at_eos=True) :
        """
        Converts an array of IDs to a sentence, optionally cutting the result
        off at the end-of-sequence token.

        :param array: 1D array containing indices
        :param cut_at_eos: cut the decoded sentences at the first <eos>
        :return: list of strings (tokens)
        """
        sentence = []
        for i in array:
            s = self.itos[i]
            if cut_at_eos and s == EOS_TOKEN:
                break
            sentence.append(s)
        return sentence

    def arrays_to_sentences(self, arrays: np.array, cut_at_eos=True):
        """
        Convert multiple arrays containing sequences of token IDs to their
        sentences, optionally cutting them off at the end-of-sequence token.

        :param arrays: 2D array containing indices
        :param cut_at_eos: cut the decoded sentences at the first <eos>
        :return: list of list of strings (tokens)
        """
        sentences = []
        for array in arrays:
            sentences.append(self.array_to_sentence(array=array, cut_at_eos=cut_at_eos))
        return sentences


if __name__ == "__main__":
    feeder = BaseFeeder()
    dataloader = torch.utils.data.DataLoader(
        dataset=feeder,
        batch_size=1,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )
    for data in dataloader:
        pdb.set_trace()
