# 残差结构，sign-encoder的结果用ctc
# teacher1: 对话上文+视频  
# teacher2: 手语翻译的句子
import pdb
import copy
from re import L
import utils
import torch
import types
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from modules.tconv import TemporalConv
from modules import BiLSTMLayer
from modules.criterions import SeqKD


from pytorch_pretrained_bert.modeling import BertConfig, BertEmbeddings,BertForSignOnly,BertNoEmbedding, BertHybridFusion
from modules.transformer import TransformerEncoder
from modules.vac import VACModel

from signjoey.decoders import Decoder, RecurrentDecoder, TransformerDecoder
from signjoey.search import beam_search, greedy
from signjoey.loss import XentLoss
from signjoey.helpers import tile



import random
import itertools
import math

import matplotlib
import matplotlib.pyplot as plt

# matplotlib.use('TkAgg')
matplotlib.use('Agg')
# matplotlib.use('pdf')


import matplotlib.pyplot as plt


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        # self.fc = nn.Linear(1024,768)

    def forward(self, x):
        # x = self.fc(x)
        return x


class SLRModel(nn.Module):
    def __init__(self, num_classes,c2d_type, conv_type, use_bn=False, tm_type='BiLSTM',
                 hidden_size= 768,bert_arg = None, decoder_arg = None,gloss_dict=None,loss_weights=None):
        super(SLRModel, self).__init__()
        # self.decoder = None
        self.loss = dict()
        self.criterion_init() 
        self.num_classes = num_classes - 2 # 去掉eos和sos
        self.num_classes_SOSEOS = num_classes 
        self.loss_weights = loss_weights
        # self.conv2d = getattr(models, c2d_type)(pretrained=True)
        # self.conv2d.fc = Identity()
        # self.conv1d = TemporalConv(input_size=512,
        #                            hidden_size=hidden_size,
        #                            conv_type=2,
        #                            use_bn=use_bn,
        #                            num_classes=num_classes)
        # self.temporal_model = BiLSTMLayer(rnn_type='LSTM', input_size=hidden_size, hidden_size=hidden_size,
        #                                       num_layers=2, bidirectional=True)
        self.gloss_dict=gloss_dict
        self.gloss_dict_word2index=dict((v, k) for k, v in gloss_dict.items())

        self.sign_encoder = VACModel(self.num_classes,c2d_type, conv_type, use_bn=use_bn, tm_type=tm_type,
                 hidden_size= hidden_size,gloss_dict=gloss_dict)
        self.sign_encoder_2 = VACModel(self.num_classes,c2d_type, conv_type, use_bn=use_bn, tm_type=tm_type,
                 hidden_size= hidden_size,gloss_dict=gloss_dict)

        self.decoder_combine = utils.Decode(gloss_dict, self.num_classes, 'beam')
        self.classifier_combine = nn.Linear(hidden_size, self.num_classes)
        self.classifier_sign = nn.Linear(hidden_size, self.num_classes)

        # if(bert_arg["model_recover_path"] is None):
        #     model_recover = None
        #     # 不读取预训练bert权重，从零开始
        #     # model_recover = {}
        # else:
        #     model_recover = torch.load(bert_arg['model_recover_path'])
            # global_step = 0

        # state_dict为None是读取预训练模型 为{}是不读取
        # self.bert_sign = BertForSignOnly.from_pretrained(
        #     bert_arg['bert_model'], state_dict={}, num_labels=2,
        #     type_vocab_size=bert_arg['type_vocab_size'], relax_projection=bert_arg['relax_projection'],
        #     config_path=bert_arg['config_path'], task_idx=bert_arg['task_idx_proj'],
        #     max_position_embeddings=bert_arg['max_position_embeddings'], label_smoothing=bert_arg['label_smoothing'],
        #     fp32_embedding=bert_arg['fp32_embedding'],
        #     cache_dir=bert_arg['output_dir'] + '/.pretrained_model_{}'.format(bert_arg['global_rank']),
        #     drop_prob=bert_arg['drop_prob'], enable_butd=bert_arg['enable_butd'],
        #     len_vis_input=bert_arg['len_vis_input'], visdial_v=bert_arg['visdial_v'], loss_type=bert_arg['loss_type'],
        #     neg_num=bert_arg['neg_num'], adaptive_weight=bert_arg['adaptive_weight'], add_attn_fuse=bert_arg['add_attn_fuse'],
        #     no_h0=bert_arg['no_h0'], no_vision=bert_arg['no_vision'],num_hidden_layers=bert_arg['num_hidden_layers_sign'],
        #     use_high_layer = None)

        self.bert_text = BertForSignOnly.from_pretrained(
            bert_arg['bert_model'], state_dict=None, num_labels=2,
            type_vocab_size=bert_arg['type_vocab_size'], relax_projection=bert_arg['relax_projection'],
            config_path=bert_arg['config_path'], task_idx=bert_arg['task_idx_proj'],
            max_position_embeddings=bert_arg['max_position_embeddings'], label_smoothing=bert_arg['label_smoothing'],
            fp32_embedding=bert_arg['fp32_embedding'],
            cache_dir=bert_arg['output_dir'] + '/.pretrained_model_{}'.format(bert_arg['global_rank']),
            drop_prob=bert_arg['drop_prob'], enable_butd=bert_arg['enable_butd'],
            len_vis_input=bert_arg['len_vis_input'], visdial_v=bert_arg['visdial_v'], loss_type=bert_arg['loss_type'],
            neg_num=bert_arg['neg_num'], adaptive_weight=bert_arg['adaptive_weight'], add_attn_fuse=bert_arg['add_attn_fuse'],
            no_h0=bert_arg['no_h0'], no_vision=bert_arg['no_vision'],num_hidden_layers=bert_arg['num_hidden_layers_text'],
            use_high_layer = None)
        
        self.bert_translation2gloss = BertForSignOnly.from_pretrained(
            bert_arg['bert_model'], state_dict=None, num_labels=2,
            type_vocab_size=bert_arg['type_vocab_size'], relax_projection=bert_arg['relax_projection'],
            config_path=bert_arg['config_path'], task_idx=bert_arg['task_idx_proj'],
            max_position_embeddings=bert_arg['max_position_embeddings'], label_smoothing=bert_arg['label_smoothing'],
            fp32_embedding=bert_arg['fp32_embedding'],
            cache_dir=bert_arg['output_dir'] + '/.pretrained_model_{}'.format(bert_arg['global_rank']),
            drop_prob=bert_arg['drop_prob'], enable_butd=bert_arg['enable_butd'],
            len_vis_input=bert_arg['len_vis_input'], visdial_v=bert_arg['visdial_v'], loss_type=bert_arg['loss_type'],
            neg_num=bert_arg['neg_num'], adaptive_weight=bert_arg['adaptive_weight'], add_attn_fuse=bert_arg['add_attn_fuse'],
            no_h0=bert_arg['no_h0'], no_vision=bert_arg['no_vision'],num_hidden_layers=bert_arg['num_hidden_layers_translation2gloss'],
            use_high_layer = None)
        # for name,params in self.bert_translation2gloss.bert.embeddings.named_parameters():
        #     # if 'sign_encoder' in name:
        #     params.requires_grad = False
        self.multimodal_fusion = BertHybridFusion.from_pretrained(
            bert_arg['bert_model'], state_dict={}, num_labels=2,
            type_vocab_size=bert_arg['type_vocab_size'], relax_projection=bert_arg['relax_projection'],
            config_path=bert_arg['config_path'], task_idx=bert_arg['task_idx_proj'],
            max_position_embeddings=bert_arg['max_position_embeddings'], label_smoothing=bert_arg['label_smoothing'],
            fp32_embedding=bert_arg['fp32_embedding'],
            cache_dir=bert_arg['output_dir'] + '/.pretrained_model_{}'.format(bert_arg['global_rank']),
            drop_prob=bert_arg['drop_prob'], enable_butd=bert_arg['enable_butd'],
            len_vis_input=bert_arg['len_vis_input'], visdial_v=bert_arg['visdial_v'], loss_type="ctc_nsp",
            neg_num=bert_arg['neg_num'], adaptive_weight=bert_arg['adaptive_weight'], add_attn_fuse=bert_arg['add_attn_fuse'],
            no_h0=bert_arg['no_h0'], no_vision=bert_arg['no_vision'],num_hidden_layers=bert_arg['num_hidden_layers_fusion'],
            use_high_layer = 6)
            
        self.multimodal_fusion.cls = self.bert_text.cls

        bert_embedding_config = BertConfig(self.num_classes_SOSEOS)
        # bert_embedding_config = {'vocab_size':137,'hidden_size':768,'max_position_embeddings':512,'type_vocab_size':2,'hidden_dropout_prob':0.1}
        self.text_embedding = BertEmbeddings(bert_embedding_config)
        self.decoder = TransformerDecoder(
                num_layers = decoder_arg['num_layers'],
                num_heads = decoder_arg['num_heads'],
                hidden_size = decoder_arg['hidden_size'],
                ff_size = decoder_arg['ff_size'],
                dropout = decoder_arg['dropout'],
                # emb_dropout = slt_arg['emb_dropout'],
                vocab_size = self.num_classes_SOSEOS,
                # freeze = slt_arg['freeze'],
            )
        self.signToText = nn.Linear(hidden_size,hidden_size)
        self.signToFusion = nn.Linear(hidden_size,hidden_size)

        self.register_backward_hook(self.backward_hook)

    def backward_hook(self, module, grad_input, grad_output):
        for g in grad_input:
            g[g != g] = 0

    def masked_bn(self, inputs, len_x):
        def pad(tensor, length):
            return torch.cat([tensor, tensor.new(length - tensor.size(0), *tensor.size()[1:]).zero_()])

        x = torch.cat([inputs[max(len_x) * idx:max(len_x) * idx + lgt] for idx, lgt in enumerate(len_x)])
        x = self.conv2d(x)
        x = torch.cat([pad(x[sum(len_x[:idx]):sum(len_x[:idx + 1])], max(len_x))
                       for idx, lgt in enumerate(len_x)])

        # x = torch.cat([inputs[len_x[0] * idx:len_x[0] * idx + lgt] for idx, lgt in enumerate(len_x)])
        # x = self.conv2d(x)
        # x = torch.cat([pad(x[sum(len_x[:idx]):sum(len_x[:idx + 1])], len_x[0])
        #                for idx, lgt in enumerate(len_x)])
        return x

    def cosine_similarity_print(self,a,figure_name):
        length_a,dim=a.size()
        cosine_similarity_image=torch.zeros(length_a,length_a)
        for j in range(0,length_a):
            for k in range(0,length_a):
                cosine_similarity_image[j,k]=torch.cosine_similarity(a[j],a[k],dim=0)
        cosine_similarity_image_np=cosine_similarity_image.cpu().detach().numpy()
        plt.figure()    
        # plt.imshow(cosine_similarity_image_np[i])
        plt.colorbar()
        ax=plt.gca()
        ax.xaxis.set_ticks_position('top')
        ax.spines['bottom'].set_position(('data',0))
        plt.savefig(figure_name)
        plt.close()
        print('')

    def sim_matrix(self,a, b, eps=1e-8):
        """
        added eps for numerical stability
        """
        a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
        a_norm = a / torch.clamp(a_n, min=eps)
        b_norm = b / torch.clamp(b_n, min=eps)
        sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
        return sim_mt

    def cosine_similarity_onevideo(self, frames, input_lengths,right_pad=None, bound=20):
        """
        input:
        a: tensor[length,dim] one video
        b: tensor[length,dim] another video which size is same as a, often a=b
        input_lengths: tensor[] size=(1) the real length of video exclude empty frame which generated by collect_fn
        bound: int  避免无手的空白帧超过一定帧数

        output:
        hand_start:动作起始帧
        hand_end:动作结束帧

        """
        # print('start')
        # print(datetime.datetime.now())
        a=frames
        b=frames
        length_a, dim = a.size()
        length_b, dim = b.size()
        cosine_similarity_image = self.sim_matrix(a, b)
        
        # 前六帧是重复的第一帧
        hand_start=0
        # hand_start =0
        # hand_end = input_lengths.item()
        hand_end = input_lengths.item()-1

        for i in range(0, input_lengths.item()):
            square_similarity = cosine_similarity_image[0:i, 0:i]
            if_no_hand = torch.gt(square_similarity, 0.95)
            if if_no_hand.all():
                hand_start += 1
            else:
                break
        for i in range(input_lengths.item()-1, 0, -1):
            square_similarity = cosine_similarity_image[i:input_lengths.item(), i:input_lengths.item()]
            if_no_hand = torch.gt(square_similarity, 0.95)
            if if_no_hand.all():
                hand_end -= 1
            else:
                break
        # 为避免得到的动作起始帧出大错，人为约束前后空白帧不超过xx帧
        hand_start = min(hand_start, bound)
        hand_end = max(hand_end, input_lengths.item() - bound)
        if hand_end-hand_start<30:
            hand_end=input_lengths.item()-1
            if hand_end-hand_start<30:
                hand_start=0
        # print('hand start{}  hand end{}  length of entity{}'.format(hand_start, hand_end, input_lengths.item()))
        return hand_start, hand_end



    def get_attn_mask(self, one_len, max_len):
        # self-attention mask
        input_mask = torch.zeros(max_len, max_len, dtype=torch.long)
        input_mask[:, :one_len].fill_(1)
        return input_mask

    def get_attn_mask_multipart(self, len_v, max_len_v, len_q, max_len_q):
        # self-attention mask
        input_mask = torch.zeros(max_len_q + max_len_v, max_len_q + max_len_v, dtype=torch.long)

        # 输入为[CLS] TEXT [SEP] VIDEO  但处理后的text中包括了[CLS][SEP]
        input_mask[:, 0: len_q].fill_(1)
        input_mask[:, max_len_q: max_len_q + len_v].fill_(1)
        return input_mask

    def get_attn_mask_cross(self, len_v, max_len_v, len_q, max_len_q):
        mask = torch.zeros(max_len_v, max_len_q, dtype=torch.long)
        mask[:, 0: len_q].fill_(1)
        return mask
    
    def get_attn_mask_boolean(self, one_len, max_len):
        # self-attention mask
        input_mask = torch.zeros(max_len, max_len, dtype=torch.bool)
        input_mask[:, :one_len].fill_(True)
        return input_mask 
    
    def forward_train(self, x, len_x_all, label, label_lgt, text, text_lgt,next_sentence_label):
            # videos
        batch, temp, channel, height, width = x.shape
        res_dict = self.sign_encoder(x, len_x_all)
        lgt = res_dict['feat_len']
        conv_logits = res_dict['conv_logits']
        sequence_logits_sign = res_dict['sequence_logits_sign']
        sign_feat = res_dict['sign_feat']

       # 获取表示手语视频帧位置的特征
        position_ids = torch.arange(
                max(lgt) , dtype=torch.long, device=sequence_logits_sign.device)
        position_ids = position_ids.expand(batch,max(lgt) )

        # 获取mask以遮蔽被补背景帧的帧和被补0的文本
        mask_sign = torch.zeros(batch,max(lgt), max(lgt) , dtype=torch.long,device=sequence_logits_sign.device)
        for i in range(batch):
            mask_sign[i] = self.get_attn_mask(lgt[i],max(lgt))

        mask_text = torch.zeros(batch,max(text_lgt), max(text_lgt) , dtype=torch.long,device=text.device)
        for i in range(batch):
            mask_text[i] = self.get_attn_mask(text_lgt[i],max(text_lgt))

        # mask = torch.zeros(batch,max(lgt) + max(text_lgt), max(lgt) + max(text_lgt), dtype=torch.long,device=sequence_logits_sign.device)
        # for i in range(batch):
        #     mask[i] = self.get_attn_mask_multipart(lgt[i],max(lgt),text_lgt[i],max(text_lgt))
        mask_VasQ = torch.zeros(batch, max(lgt), max(text_lgt) , dtype=torch.long,device=text.device)
        for i in range(batch):
            mask_VasQ[i] = self.get_attn_mask_cross(lgt[i], max(lgt), text_lgt[i], max(text_lgt))

        bert_output_text = self.bert_text(vis_feats=None,vis_pe=None,input_ids=text,attention_mask=mask_text)
        # outputs_sign = self.classifier_sign(bert_output_sign_video.transpose(0,1))
        pred_sign = self.decoder_combine.decode(sequence_logits_sign, lgt, batch_first=False, probs=False)
        
        bert_output_fusion_list,next_sentence_loss,seq_relationship_score = self.multimodal_fusion(
            vis_feat = sign_feat.transpose(0,1),
            text_feat = bert_output_text,
            position_ids=position_ids,
            attention_mask_vis = mask_sign,
            attention_mask_text = mask_text,
            attention_mask_VasQ = mask_VasQ,
            next_sentence_label=next_sentence_label)
        

        bert_output_classifier = bert_output_fusion_list[0]
        bert_output_classifier = bert_output_classifier.transpose(0,1)
        sequence_logits = self.classifier_combine(bert_output_classifier)
        # greedyPred = self.decoder_combine.decode(outputs, lgt, batch_first=False, probs=False,search_mode='max')

        pred = self.decoder_combine.decode(sequence_logits, lgt, batch_first=False, probs=False)

        return {
            # "framewise_features": bert_output_fusion,
            # "feat_len": lgt,
            # "conv_logits" : conv_logits,
            "bert_output_classifier": bert_output_classifier,
            "sequence_logits_teacher": sequence_logits,
            # "sequence_logits_sign" : sequence_logits_sign,
            # "recognized_sents": pred,
            # "recognized_sents_sign": pred_sign,
            # "next_sentence_loss": next_sentence_loss
        }
    def forward_train_translation2gloss(self, label_with_SOS, label_with_SOS_lgt, text_translation, text_translation_lgt):
        # 输入不包括<eos>
        batch = label_with_SOS.size(0)
        unroll_steps = label_with_SOS.size(1)
        trg_embed = self.text_embedding(vis_feats=None,vis_pe=None,input_ids=label_with_SOS)

        mask_text = torch.zeros(batch,max(text_translation_lgt), max(text_translation_lgt) , dtype=torch.long,device=label_with_SOS.device)
        for i in range(batch):
            mask_text[i] = self.get_attn_mask(text_translation_lgt[i],max(text_translation_lgt))

        trg_mask = torch.zeros(batch,max(label_with_SOS_lgt), max(label_with_SOS_lgt) , dtype=torch.bool,device=label_with_SOS.device)
        for i in range(batch):
            trg_mask[i] = self.get_attn_mask_boolean(label_with_SOS_lgt[i],max(label_with_SOS_lgt))

        src_mask = torch.zeros(batch,max(text_translation_lgt), max(text_translation_lgt) , dtype=torch.bool,device=label_with_SOS.device)
        for i in range(batch):
            src_mask[i] = self.get_attn_mask_boolean(text_translation_lgt[i],max(text_translation_lgt))
        
        # src_mask = trg_mask
        encoder_output = self.bert_translation2gloss(vis_feats=None,vis_pe=None,input_ids=text_translation,attention_mask=mask_text)
        decoder_outputs = self.decoder(
            trg_embed=trg_embed, 
            encoder_output=encoder_output,
            src_mask=src_mask[:,0:1,:],
            trg_mask=trg_mask,        
                # hidden=decoder_hidden,
        )
        word_outputs, word_feature, _, _ = decoder_outputs
        # Calculate Translation Loss
        txt_log_probs = F.log_softmax(word_outputs, dim=-1)

        return {
            "txt_log_probs": txt_log_probs,
            "word_outputs": word_outputs,
            "word_feature": word_feature
        }

    def forward_train_student(self, x, len_x_all, label, label_lgt, text, text_lgt,next_sentence_label):
            # videos

        res_dict_2 = self.sign_encoder_2(x, len_x_all)
        lgt_2 = res_dict_2['feat_len']
        conv_logits_2 = res_dict_2['conv_logits']
        sequence_logits_sign_2 = res_dict_2['sequence_logits_sign']
        sign_feat_student = self.signToText(res_dict_2['sign_feat'])
        sign_feat_student_fusion = self.signToFusion(res_dict_2['sign_feat'])

        pred_sign,peak_timestep = self.decoder_combine.decode(sequence_logits_sign_2, lgt_2, batch_first=False, probs=False,search_mode='peak')
        pred_notext = self.decoder_combine.decode(sequence_logits_sign_2, lgt_2, batch_first=False, probs=False,search_mode='no_text')
        # sequence_logits_teacher = sequence_logits_teacher
        # print(pred_sign)
        sequence_logits_sign_softmax = sequence_logits_sign_2.softmax(-1)
        # for glossindex,trueindex in enumerate(pred_notext[0]):
        #     prob_one = sequence_logits_sign_softmax[:,0,trueindex]
        #     x1 = []
        #     y1 = []
        #     for i in range(0,prob_one.size(0)):
        #         x1.append(i)
        #         y1.append(prob_one[i].item())
        
        #     plt.plot(x1,y1,label = pred_sign[0][glossindex][0])
        # plt.xlabel('Timestep')
        # plt.ylabel('Probability')
        # plt.legend(bbox_to_anchor=(0.8, 0.6), loc=3, borderaxespad=0)
        # plt.savefig("ctc_peak2.png")
        # plt.show()
        # plt.close()



        return {
            # "framewise_features": bert_output_fusion,
            "feat_len": lgt_2,
            "conv_logits" : conv_logits_2,
            # "sequence_logits_teacher": sequence_logits_teacher,
            "sign_feat_student": sign_feat_student,
            "sign_feat_student_fusion": sign_feat_student_fusion,
            "sequence_logits_sign" : sequence_logits_sign_2,
            "peak_timestep" : peak_timestep
            # "recognized_sents": pred,
            # "recognized_sents_sign": pred_sign,
            # "next_sentence_loss": next_sentence_loss
        }

    def forward_test(self, x, len_x_all, label, label_lgt, text, text_lgt):
        batch, temp, channel, height, width = x.shape

        res_dict = self.sign_encoder_2(x, len_x_all)
        lgt = res_dict['feat_len']
        conv_logits = res_dict['conv_logits']
        sequence_logits_sign = res_dict['sequence_logits_sign']
        sign_feat = res_dict['sign_feat']

        pred_sign = self.decoder_combine.decode(sequence_logits_sign, lgt, batch_first=False, probs=False)
        
        return {
            "feat_len": lgt,
            "sequence_logits_sign" : sequence_logits_sign,
            "recognized_sents": pred_sign,
        }

    def criterion_calculation(self, ret_dict_teacher,ret_dict_teacher_translation2gloss, ret_dict_student,label, label_lgt, label_with_EOS,epoch,loss_weights):
        loss = 0
        
        # E_label_lgt=torch.ones(4)
        
        loss_ctc_RNN = torch.tensor(0).cuda().float()
        loss_ctc_conv1d = torch.tensor(0).cuda().float()
        loss_dist_RNN_to_conv1d = torch.tensor(0).cuda().float()
        loss_dist_fusion_to_slr = torch.tensor(0).cuda().float()
        loss_dist_fusion_to_slr_feature = torch.tensor(0).cuda().float()
        loss_dist_translation_to_slr = torch.tensor(0).cuda().float()
        loss_dist_translation_to_slr_feature = torch.tensor(0).cuda().float()
        for k, weight in loss_weights.items():
            if k == 'ConvCTCSign':
                loss_ctc_conv1d = self.loss['CTCLoss'](ret_dict_student["conv_logits"].log_softmax(-1),
                                                      label.cpu().int(), ret_dict_student["feat_len"].cpu().int(),
                                                      label_lgt.cpu().int()).mean()
                loss += weight * loss_ctc_conv1d
            elif k == 'SeqCTCSign':
                # if epoch>2:
                loss_ctc_RNN = self.loss['CTCLoss'](ret_dict_student["sequence_logits_sign"].log_softmax(-1),
                                                      label.cpu().int(), ret_dict_student["feat_len"].cpu().int(),
                                                      label_lgt.cpu().int()).mean()
                loss += weight * loss_ctc_RNN
            elif k == 'Dist':
                loss_dist_RNN_to_conv1d = self.loss['distillation'](ret_dict_student["conv_logits"],
                                                           ret_dict_student["sequence_logits_sign"].detach(),
                                                           use_blank=False)
                loss += weight * loss_dist_RNN_to_conv1d
            # elif k == 'next_sentence_loss':
            #     # if epoch > 10:
            #     loss += weight * ret_dict["next_sentence_loss"].mean()
            # elif k == 'SeqCTC':
            #     # if epoch > 2:
            #     loss += weight * self.loss['CTCLoss'](ret_dict["sequence_logits"].log_softmax(-1),
            #                                           label.cpu().int(), ret_dict["feat_len"].cpu().int(),
            #                                           label_lgt.cpu().int()).mean()
            elif k == 'FeatureFusion2SLR':
                loss_dist_fusion_to_slr_feature = self.loss['distillation_feature'](ret_dict_student["sign_feat_student_fusion"],
                                                                ret_dict_teacher["bert_output_classifier"].detach(),
                                                                )
                loss += weight * loss_dist_fusion_to_slr_feature
            elif k == 'DistFusion2SLR':
                loss_dist_fusion_to_slr = self.loss['distillation'](ret_dict_student["sequence_logits_sign"],
                                                           ret_dict_teacher["sequence_logits_teacher"].detach(),
                                                           use_blank=False)
                loss += weight * loss_dist_fusion_to_slr
            elif k == 'DistTranslation2SLR':
                ifBackward = True
                peak_timestep = ret_dict_student['peak_timestep']
                batch = peak_timestep.size(0)
                peak_timestep = peak_timestep.cuda()        
                peak_timestep = peak_timestep.type(torch.int64)
                # 第一个维度（时序）上去掉<EOS>
                # 第三个维度（词典）上去掉<EOS>和<SOS>
                _ ,greedy_gloss = torch.max(ret_dict_teacher_translation2gloss["word_outputs"] ,dim = -1)
                for i in range(0,batch):
                    teachar_used_length = greedy_gloss.size(1)
                    for j in range(greedy_gloss.size(1)):
                        if greedy_gloss[i][j] == 1117:
                            teachar_used_length = j
                            break
                    teacher_logits = ret_dict_teacher_translation2gloss["word_outputs"].transpose(0,1)[:teachar_used_length,i,:-2].detach() 
                    student_used_length = peak_timestep.size(1)
                    for j in range(1,peak_timestep.size(1)):
                        if peak_timestep[i][j] == 0:
                            student_used_length = j
                            break
                    peak_timestep_one = peak_timestep[i,:student_used_length]
                    student_logits = torch.gather(ret_dict_student["sequence_logits_sign"][:,i],0,peak_timestep_one.unsqueeze(1).expand(-1, ret_dict_student["sequence_logits_sign"].size(-1)))
                    student_length = student_logits.size(0)
                    teacher_length = teacher_logits.size(0)
                    if student_length > teacher_length:
                        student_logits_use = student_logits[:teacher_length]
                        if student_length > (teacher_length * 2):
                            ifBackward = False
                            break
                    elif student_length < teacher_length:
                        try:
                            student_logits_use = torch.cat(
                                (
                                    student_logits,
                                    student_logits[-1].expand(teacher_length - student_length, -1),
                                )
                            , dim=0)
                        except:
                            # print(student_logits.size())
                            # print(ret_dict_student["sequence_logits_sign"].size())
                            # print(peak_timestep)
                            ifBackward = False
                            break
                    else:
                        student_logits_use = student_logits
                    loss_dist_translation_to_slr += self.loss['distillation'](student_logits_use.unsqueeze(1),
                                                                teacher_logits.unsqueeze(1),
                                                                use_blank=False)
                
                loss += weight * loss_dist_translation_to_slr
            
            elif k == 'FeatureTranslation2SLR':
                peak_timestep = ret_dict_student['peak_timestep']
                batch = peak_timestep.size(0)
                peak_timestep = peak_timestep.cuda()        
                peak_timestep = peak_timestep.type(torch.int64)
                # 第一个维度（时序）上去掉<EOS>
                # 由于是特征，所以不需要第三个维度（词典）上去掉<EOS>和<SOS>
                _ ,greedy_gloss = torch.max(ret_dict_teacher_translation2gloss["word_outputs"] ,dim = -1)
                for i in range(0,batch):
                    teachar_used_length = greedy_gloss.size(1)
                    for j in range(greedy_gloss.size(1)):
                        if greedy_gloss[i][j] == 1117:
                            teachar_used_length = j
                            break
                    teacher_feature = ret_dict_teacher_translation2gloss["word_feature"].transpose(0,1)[:teachar_used_length,i].detach() 
                    student_used_length = peak_timestep.size(1)
                    for j in range(1,peak_timestep.size(1)):
                        if peak_timestep[i][j] == 0:
                            student_used_length = j
                            break
                    peak_timestep_one = peak_timestep[i,:student_used_length]
                    student_feature = torch.gather(ret_dict_student["sign_feat_student"][:,i],0,peak_timestep_one.unsqueeze(1).expand(-1, ret_dict_student["sign_feat_student"].size(-1)))
                    student_length = student_feature.size(0)
                    teacher_length = teacher_feature.size(0)
                    if student_length > teacher_length:
                        student_feature_use = student_feature[:teacher_length]
                        if student_length > (teacher_length * 2):
                            ifBackward = False
                            break
                    elif student_length < teacher_length:
                        try:
                            student_feature_use = torch.cat(
                                (
                                    student_feature,
                                    student_feature[-1].expand(teacher_length - student_length, -1),
                                )
                            , dim=0)
                        except:
                            # print(student_logits.size())
                            # print(ret_dict_student["sequence_logits_sign"].size())
                            # print(peak_timestep)
                            ifBackward = False
                            break
                    else:
                        student_feature_use = student_feature
                    loss_dist_translation_to_slr_feature += self.loss['distillation_feature'](student_feature_use,
                                                                teacher_feature,
                                                                )
                
                loss += weight * loss_dist_translation_to_slr_feature
                    

            # elif k == 'DistTranslation2SLR':
            #     peak_timestep = ret_dict_student['peak_timestep']
            #     peak_timestep = peak_timestep.cuda()
                
            #     # peak_timestep = torch.tensor(peak_timestep, dtype=torch.int64)
            #     peak_timestep = peak_timestep.type(torch.int64)
            #     peak_timestep = peak_timestep.transpose(0,1)
            #     student_logits = torch.gather(ret_dict_student["sequence_logits_sign"],0,peak_timestep.unsqueeze(2).expand(-1, -1, ret_dict_student["sequence_logits_sign"].size(-1)))
            #     # 第一个维度（时序）上去掉<EOS>
            #     # 第三个维度（词典）上去掉<EOS>和<SOS>
            #     teacher_logits = ret_dict_teacher_translation2gloss["word_outputs"].transpose(0,1)[:-1,:,:-2].detach() 
            #     student_length = student_logits.size(0)
            #     teacher_length = teacher_logits.size(0)
            #     if student_length > teacher_length:
            #         student_logits_use = student_logits[:teacher_length]
            #     elif student_length < teacher_length:
            #         try:
            #             student_logits_use = torch.cat(
            #                 (
            #                     student_logits,
            #                     student_logits[-1].expand(teacher_length - student_length, -1, -1),
            #                 )
            #             , dim=0)
            #         except:
            #             print(student_logits.size())
            #             print(ret_dict_student["sequence_logits_sign"].size())
            #             print(peak_timestep)
            #             continue
            #     else:
            #         student_logits_use = student_logits
            #     loss_dist_translation_to_slr = self.loss['distillation'](student_logits_use,
            #                                                teacher_logits,
            #                                                use_blank=False)
            #     loss += weight * loss_dist_fusion_to_slr
            
            elif k == 'TranslationToGlossCrossEntropy':
                loss += weight * self.loss['translation'](
                    ret_dict_teacher_translation2gloss['txt_log_probs'], label_with_EOS,
                )
            
            

            
            
            

        return loss,loss_ctc_RNN,loss_ctc_conv1d,loss_dist_RNN_to_conv1d,loss_dist_fusion_to_slr,loss_dist_translation_to_slr

    def criterion_init(self):
        self.loss['CTCLoss'] = torch.nn.CTCLoss(reduction='none', zero_infinity=False)
        self.loss['distillation'] = SeqKD(T=8)
        self.loss['translation'] = XentLoss(
            pad_index=0, smoothing=0.0
        )
        self.loss['distillation_feature'] = torch.nn.MSELoss(reduction='mean')
        # self.loss['distillation_disentangle'] = SeqKD(T=8)
        return self.loss
