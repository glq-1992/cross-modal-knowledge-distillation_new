import os
import pdb
import time
import torch
import ctcdecode
import numpy as np
from itertools import groupby
import torch.nn.functional as F


class Decode(object):
    def __init__(self, gloss_dict, num_classes, search_mode, blank_id=0):
        self.i2g_dict = dict((v, k) for k, v in gloss_dict.items())

        # 新增blank，为了分词
        self.i2g_dict[0]='blank'

        self.g2i_dict = {v: k for k, v in self.i2g_dict.items()}
        self.num_classes = num_classes
        self.search_mode = search_mode
        self.blank_id = blank_id
        vocab = [chr(x) for x in range(20000, 20000 + num_classes)]
        self.ctc_decoder = ctcdecode.CTCBeamDecoder(vocab, beam_width=10, blank_id=blank_id,
                                                    num_processes=10)

    def decode(self, nn_output, vid_lgt, batch_first=True, probs=False,search_mode='Beam'):
        if not batch_first:
            nn_output = nn_output.permute(1, 0, 2)
        if search_mode == "max":
            return self.MaxDecode(nn_output, vid_lgt)
        elif search_mode == "no_text":
            return self.BeamSearch_no_text(nn_output, vid_lgt, probs)
        elif search_mode == 'peak':
            return self.BeamSearchPeak(nn_output, vid_lgt, probs)
        else:
            return self.BeamSearch(nn_output, vid_lgt, probs)

    def BeamSearch(self, nn_output, vid_lgt, probs=False):
        '''
        CTCBeamDecoder Shape:
                - Input:  nn_output (B, T, N), which should be passed through a softmax layer
                - Output: beam_resuls (B, N_beams, T), int, need to be decoded by i2g_dict
                          beam_scores (B, N_beams), p=1/np.exp(beam_score)
                          timesteps (B, N_beams)
                          out_lens (B, N_beams)
        '''
        if not probs:
            nn_output = nn_output.softmax(-1).cpu()
        vid_lgt = vid_lgt.cpu()
        beam_result, beam_scores, timesteps, out_seq_len = self.ctc_decoder.decode(nn_output, vid_lgt)
        ret_list = []
        for batch_idx in range(len(nn_output)):
            first_result = beam_result[batch_idx][0][:out_seq_len[batch_idx][0]]
            if len(first_result) != 0:
                first_result = torch.stack([x[0] for x in groupby(first_result)])
            try:
                ret_list.append([(self.i2g_dict[int(gloss_id)], idx) for idx, gloss_id in
                             enumerate(first_result)])
            except:
                pass
        return ret_list

    def BeamSearchPeak(self, nn_output, vid_lgt, probs=False):
        '''
        CTCBeamDecoder Shape:
                - Input:  nn_output (B, T, N), which should be passed through a softmax layer
                - Output: beam_resuls (B, N_beams, T), int, need to be decoded by i2g_dict
                          beam_scores (B, N_beams), p=1/np.exp(beam_score)
                          timesteps (B, N_beams)
                          out_lens (B, N_beams)
        '''
        if not probs:
            nn_output = nn_output.softmax(-1).cpu()
        vid_lgt = vid_lgt.cpu()
        beam_result, beam_scores, timesteps, out_seq_len = self.ctc_decoder.decode(nn_output, vid_lgt)
        ret_list = []
        timestepsList = []
        max_decode_length = 0
        for batch_idx in range(len(nn_output)):
            first_result = beam_result[batch_idx][0][:out_seq_len[batch_idx][0]]
            max_decode_length = max(max_decode_length, out_seq_len[batch_idx][0])
            if len(first_result) != 0:
                first_result = torch.stack([x[0] for x in groupby(first_result)])
            try:
                ret_list.append([(self.i2g_dict[int(gloss_id)], idx) for idx, gloss_id in
                             enumerate(first_result)])
                timestepsList.append(timesteps[batch_idx][0][:out_seq_len[batch_idx][0]])
            except:
                pass
        timestepsTensor = torch.zeros(len(nn_output), max_decode_length)
        for batch_idx in range(len(nn_output)):
            timestepsTensor[batch_idx,:out_seq_len[batch_idx][0]] = timestepsList[batch_idx]
        return ret_list, timestepsTensor


    def BeamSearch_no_text(self, nn_output, vid_lgt, probs=False):
        '''
        CTCBeamDecoder Shape:
                - Input:  nn_output (B, T, N), which should be passed through a softmax layer
                - Output: beam_resuls (B, N_beams, T), int, need to be decoded by i2g_dict
                          beam_scores (B, N_beams), p=1/np.exp(beam_score)
                          timesteps (B, N_beams)
                          out_lens (B, N_beams)
        '''
        if not probs:
            nn_output = nn_output.softmax(-1).cpu()
        vid_lgt = vid_lgt.cpu()
        beam_result, beam_scores, timesteps, out_seq_len = self.ctc_decoder.decode(nn_output, vid_lgt)
        ret_list = []
        for batch_idx in range(len(nn_output)):
            first_result = beam_result[batch_idx][0][:out_seq_len[batch_idx][0]]
            if len(first_result) != 0:
                first_result = torch.stack([x[0] for x in groupby(first_result)])
            try:
                ret_list.append([(int(gloss_id)) for idx, gloss_id in
                             enumerate(first_result)])
            except:
                pass
        return ret_list

    # def MaxDecode(self, nn_output, vid_lgt):
    #     index_list = torch.argmax(nn_output, axis=2)
    #     batchsize, lgt = index_list.shape
    #     ret_list = []
    #     for batch_idx in range(batchsize):
    #         group_result = [x[0] for x in groupby(index_list[batch_idx][:vid_lgt[batch_idx]])]
    #         filtered = [*filter(lambda x: x != self.blank_id, group_result)]
    #         if len(filtered) > 0:
    #             max_result = torch.stack(filtered)
    #             max_result = [x[0] for x in groupby(max_result)]
    #         else:
    #             max_result = filtered
    #         ret_list.append([(self.i2g_dict[int(gloss_id)], idx) for idx, gloss_id in
    #                          enumerate(max_result)])
    #     return ret_list
    # 不去除blank的版本
    def MaxDecode(self, nn_output, vid_lgt):
        index_list = torch.argmax(nn_output, axis=2)
        batchsize, lgt = index_list.shape
        ret_list = []
        for batch_idx in range(batchsize):
            group_result = [x for x in (index_list[batch_idx][:vid_lgt[batch_idx]])]
            # filtered = [*filter(lambda x: x != self.blank_id, group_result)]
            if len(group_result) > 0:
                max_result = torch.stack(group_result)
                max_result = [x for x in max_result]
            else:
                max_result = group_result
            # 返回预测的词、词id、第几个词
            try:
                ret_list.append([(self.i2g_dict[int(gloss_id)],int(gloss_id), idx) for idx, gloss_id in
                             enumerate(max_result)])
            except Exception as e:
                print(e)
            finally:
                continue
        return ret_list
