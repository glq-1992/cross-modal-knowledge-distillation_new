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
from modules.vac import VACModel,VACModelNoCNN

from signjoey.decoders import Decoder, RecurrentDecoder, TransformerDecoder
from signjoey.search import beam_search, greedy
from signjoey.loss import XentLoss
from signjoey.helpers import tile



import random
import itertools
import math

import matplotlib
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
                 hidden_size= 768,bert_arg = None, gloss2text_arg = None,slt_arg = None,gloss_dict=None,text_dict = None,loss_weights=None):
        super(SLRModel, self).__init__()
        # self.decoder = None
        self.loss = dict()
        self.criterion_init() 
        self.num_classes = num_classes
        # self.num_classes_SOSEOS = num_classes 
        self.loss_weights = loss_weights
        

        self.gloss_dict=gloss_dict
        self.text_dict = text_dict
        self.gloss_dict_word2index=dict((v, k) for k, v in gloss_dict.items())

        self.sign_encoder = VACModel(self.num_classes,c2d_type, conv_type, use_bn=use_bn, tm_type=tm_type,
                 hidden_size= hidden_size,gloss_dict=gloss_dict)
        self.sign_encoder_2 = VACModel(self.num_classes,c2d_type, conv_type, use_bn=use_bn, tm_type=tm_type,
                 hidden_size= hidden_size,gloss_dict=gloss_dict)

        self.decoder_combine = utils.Decode(gloss_dict, self.num_classes, 'beam')
        self.classifier_combine = nn.Linear(hidden_size, self.num_classes)
        self.classifier_sign = nn.Linear(hidden_size, self.num_classes)


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
        
        self.bert_gloss2text = BertForSignOnly.from_pretrained(
            bert_arg['bert_model'], state_dict={}, num_labels=2,
            vocab_size = self.num_classes, # 将vocabsize改为gloss的种类
            type_vocab_size=bert_arg['type_vocab_size'], relax_projection=bert_arg['relax_projection'],
            config_path=bert_arg['config_path'], task_idx=bert_arg['task_idx_proj'],
            max_position_embeddings=bert_arg['max_position_embeddings'], label_smoothing=bert_arg['label_smoothing'],
            fp32_embedding=bert_arg['fp32_embedding'],
            cache_dir=bert_arg['output_dir'] + '/.pretrained_model_{}'.format(bert_arg['global_rank']),
            drop_prob=bert_arg['drop_prob'], enable_butd=bert_arg['enable_butd'],
            len_vis_input=bert_arg['len_vis_input'], visdial_v=bert_arg['visdial_v'], loss_type=bert_arg['loss_type'],
            neg_num=bert_arg['neg_num'], adaptive_weight=bert_arg['adaptive_weight'], add_attn_fuse=bert_arg['add_attn_fuse'],
            no_h0=bert_arg['no_h0'], no_vision=bert_arg['no_vision'],num_hidden_layers=bert_arg['num_hidden_layers_gloss2text'],
            use_high_layer = None)
        # for name,params in self.bert_gloss2text.bert.embeddings.named_parameters():
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

        bert_embedding_config = BertConfig(len(self.text_dict))
        # bert_embedding_config = {'vocab_size':137,'hidden_size':768,'max_position_embeddings':512,'type_vocab_size':2,'hidden_dropout_prob':0.1}
        self.text_embedding = BertEmbeddings(bert_embedding_config)
        self.decoder_gloss2text = TransformerDecoder(
                num_layers = gloss2text_arg['num_layers'],
                num_heads = gloss2text_arg['num_heads'],
                hidden_size = gloss2text_arg['hidden_size'],
                ff_size = gloss2text_arg['ff_size'],
                dropout = gloss2text_arg['dropout'],
                # emb_dropout = slt_arg['emb_dropout'],
                vocab_size = len(self.text_dict),
                # freeze = slt_arg['freeze'],
            )
        
        self.decoder_sign2text = TransformerDecoder(
                num_layers = slt_arg['num_layers'],
                num_heads = slt_arg['num_heads'],
                hidden_size = slt_arg['hidden_size'],
                ff_size = slt_arg['ff_size'],
                dropout = slt_arg['dropout'],
                # emb_dropout = slt_arg['emb_dropout'],
                vocab_size = len(self.text_dict),
                # freeze = slt_arg['freeze'],
            )
        self.signToText = nn.Linear(hidden_size,hidden_size)


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
    def transformer_greedy(
        src_mask,
        embed,
        bos_index: int,
        eos_index: int,
        max_output_length: int,
        decoder: Decoder,
        encoder_output,
        encoder_hidden,
    ) :
        """
        Special greedy function for transformer, since it works differently.
        The transformer remembers all previous states and attends to them.

        :param src_mask: mask for source inputs, 0 for positions after </s>
        :param embed: target embedding layer
        :param bos_index: index of <s> in the vocabulary
        :param eos_index: index of </s> in the vocabulary
        :param max_output_length: maximum length for the hypotheses
        :param decoder: decoder to use for greedy decoding
        :param encoder_output: encoder hidden states for attention
        :param encoder_hidden: encoder final state (unused in Transformer)
        :return:
            - stacked_output: output hypotheses (2d array of indices),
            - stacked_attention_scores: attention scores (3d array)
        """

        batch_size = src_mask.size(0)

        # start with BOS-symbol for each sentence in the batch
        ys = encoder_output.new_full([batch_size, 1], bos_index, dtype=torch.long)

        # a subsequent mask is intersected with this in decoder forward pass
        trg_mask = src_mask.new_ones([1, 1, 1])
        finished = src_mask.new_zeros((batch_size)).byte()

        for _ in range(max_output_length):

            trg_embed = embed(ys)  # embed the previous tokens

            # pylint: disable=unused-variable
            with torch.no_grad():
                logits, out, _, _ = decoder(
                    trg_embed=trg_embed,
                    encoder_output=encoder_output,
                    encoder_hidden=None,
                    src_mask=src_mask,
                    unroll_steps=None,
                    hidden=None,
                    trg_mask=trg_mask,
                )

                logits = logits[:, -1]
                _, next_word = torch.max(logits, dim=1)
                next_word = next_word.data
                ys = torch.cat([ys, next_word.unsqueeze(-1)], dim=1)

            # check if previous symbol was <eos>
            is_eos = torch.eq(next_word, eos_index)
            finished += is_eos
            # stop predicting if <eos> reached for all elements in batch
            if (finished >= 1).sum() == batch_size:
                break

        ys = ys[:, 1:]  # remove BOS-symbol
        return ys.detach().cpu().numpy(), None

    def beam_search(
        decoder: Decoder,
        size: int,
        bos_index: int,
        eos_index: int,
        pad_index: int,
        encoder_output,
        encoder_hidden,
        src_mask,
        max_output_length: int,
        alpha: float,
        embed,
        n_best: int = 1,
    ):
        """
        Beam search with size k.
        Inspired by OpenNMT-py, adapted for Transformer.

        In each decoding step, find the k most likely partial hypotheses.

        :param decoder:
        :param size: size of the beam
        :param bos_index:
        :param eos_index:
        :param pad_index:
        :param encoder_output:
        :param encoder_hidden:
        :param src_mask:
        :param max_output_length:
        :param alpha: `alpha` factor for length penalty
        :param embed:
        :param n_best: return this many hypotheses, <= beam (currently only 1)
        :return:
            - stacked_output: output hypotheses (2d array of indices),
            - stacked_attention_scores: attention scores (3d array)
        """
        assert size > 0, "Beam size must be >0."
        assert n_best <= size, "Can only return {} best hypotheses.".format(size)

        # init
        transformer = isinstance(decoder, TransformerDecoder)
        batch_size = src_mask.size(0)
        att_vectors = None  # not used for Transformer

        # Recurrent models only: initialize RNN hidden state
        # pylint: disable=protected-access
        if not transformer:
            hidden = decoder._init_hidden(encoder_hidden)
        else:
            hidden = None

        # tile encoder states and decoder initial states beam_size times
        if hidden is not None:
            hidden = tile(hidden, size, dim=1)  # layers x batch*k x dec_hidden_size

        encoder_output = tile(
            encoder_output.contiguous(), size, dim=0
        )  # batch*k x src_len x enc_hidden_size
        src_mask = tile(src_mask, size, dim=0)  # batch*k x 1 x src_len

        # Transformer only: create target mask
        if transformer:
            trg_mask = src_mask.new_ones([1, 1, 1])  # transformer only
        else:
            trg_mask = None

        # numbering elements in the batch
        batch_offset = torch.arange(
            batch_size, dtype=torch.long, device=encoder_output.device
        )

        # numbering elements in the extended batch, i.e. beam size copies of each
        # batch element
        beam_offset = torch.arange(
            0, batch_size * size, step=size, dtype=torch.long, device=encoder_output.device
        )

        # keeps track of the top beam size hypotheses to expand for each element
        # in the batch to be further decoded (that are still "alive")
        alive_seq = torch.full(
            [batch_size * size, 1],
            bos_index,
            dtype=torch.long,
            device=encoder_output.device,
        )

        # Give full probability to the first beam on the first step.
        topk_log_probs = torch.zeros(batch_size, size, device=encoder_output.device)
        topk_log_probs[:, 1:] = float("-inf")

        # Structure that holds finished hypotheses.
        hypotheses = [[] for _ in range(batch_size)]

        results = {
            "predictions": [[] for _ in range(batch_size)],
            "scores": [[] for _ in range(batch_size)],
            "gold_score": [0] * batch_size,
        }

        for step in range(max_output_length):

            # This decides which part of the predicted sentence we feed to the
            # decoder to make the next prediction.
            # For Transformer, we feed the complete predicted sentence so far.
            # For Recurrent models, only feed the previous target word prediction
            if transformer:  # Transformer
                decoder_input = alive_seq  # complete prediction so far
            else:  # Recurrent
                decoder_input = alive_seq[:, -1].view(-1, 1)  # only the last word

            # expand current hypotheses
            # decode one single step
            # logits: logits for final softmax
            # pylint: disable=unused-variable
            trg_embed = embed(decoder_input)
            logits, hidden, att_scores, att_vectors = decoder(
                encoder_output=encoder_output,
                encoder_hidden=encoder_hidden,
                src_mask=src_mask,
                trg_embed=trg_embed,
                hidden=hidden,
                prev_att_vector=att_vectors,
                unroll_steps=1,
                trg_mask=trg_mask,  # subsequent mask for Transformer only
            )

            # For the Transformer we made predictions for all time steps up to
            # this point, so we only want to know about the last time step.
            if transformer:
                logits = logits[:, -1]  # keep only the last time step
                hidden = None  # we don't need to keep it for transformer

            # batch*k x trg_vocab
            log_probs = F.log_softmax(logits, dim=-1).squeeze(1)

            # multiply probs by the beam probability (=add logprobs)
            log_probs += topk_log_probs.view(-1).unsqueeze(1)
            curr_scores = log_probs.clone()

            # compute length penalty
            if alpha > -1:
                length_penalty = ((5.0 + (step + 1)) / 6.0) ** alpha
                curr_scores /= length_penalty

            # flatten log_probs into a list of possibilities
            curr_scores = curr_scores.reshape(-1, size * decoder.output_size)

            # pick currently best top k hypotheses (flattened order)
            topk_scores, topk_ids = curr_scores.topk(size, dim=-1)

            if alpha > -1:
                # recover original log probs
                topk_log_probs = topk_scores * length_penalty
            else:
                topk_log_probs = topk_scores.clone()

            # reconstruct beam origin and true word ids from flattened order
            topk_beam_index = topk_ids.div(decoder.output_size)
            topk_ids = topk_ids.fmod(decoder.output_size)

            # map beam_index to batch_index in the flat representation
            batch_index = topk_beam_index + beam_offset[
                : topk_beam_index.size(0)
            ].unsqueeze(1)
            select_indices = batch_index.view(-1)

            # append latest prediction
            alive_seq = torch.cat(
                [alive_seq.index_select(0, select_indices), topk_ids.view(-1, 1)], -1
            )  # batch_size*k x hyp_len

            is_finished = topk_ids.eq(eos_index)
            if step + 1 == max_output_length:
                is_finished.fill_(True)
            # end condition is whether the top beam is finished
            end_condition = is_finished[:, 0].eq(True)

            # save finished hypotheses
            if is_finished.any():
                predictions = alive_seq.view(-1, size, alive_seq.size(-1))
                for i in range(is_finished.size(0)):
                    b = batch_offset[i]
                    if end_condition[i]:
                        is_finished[i].fill_(True)
                    finished_hyp = is_finished[i].nonzero().view(-1)
                    # store finished hypotheses for this batch
                    for j in finished_hyp:
                        # Check if the prediction has more than one EOS.
                        # If it has more than one EOS, it means that the prediction should have already
                        # been added to the hypotheses, so you don't have to add them again.
                        if (predictions[i, j, 1:] == eos_index).nonzero().numel() < 2:
                            hypotheses[b].append(
                                (
                                    topk_scores[i, j],
                                    predictions[i, j, 1:],
                                )  # ignore start_token
                            )
                    # if the batch reached the end, save the n_best hypotheses
                    if end_condition[i]:
                        best_hyp = sorted(hypotheses[b], key=lambda x: x[0], reverse=True)
                        for n, (score, pred) in enumerate(best_hyp):
                            if n >= n_best:
                                break
                            results["scores"][b].append(score)
                            results["predictions"][b].append(pred)
                non_finished = end_condition.eq(False).nonzero().view(-1)
                # if all sentences are translated, no need to go further
                # pylint: disable=len-as-condition
                if len(non_finished) == 0:
                    break
                # remove finished batches for the next step
                topk_log_probs = topk_log_probs.index_select(0, non_finished)
                batch_index = batch_index.index_select(0, non_finished)
                batch_offset = batch_offset.index_select(0, non_finished)
                alive_seq = predictions.index_select(0, non_finished).view(
                    -1, alive_seq.size(-1)
                )

            # reorder indices, outputs and masks
            select_indices = batch_index.view(-1)
            encoder_output = encoder_output.index_select(0, select_indices)
            src_mask = src_mask.index_select(0, select_indices)

            if hidden is not None and not transformer:
                if isinstance(hidden, tuple):
                    # for LSTMs, states are tuples of tensors
                    h, c = hidden
                    h = h.index_select(1, select_indices)
                    c = c.index_select(1, select_indices)
                    hidden = (h, c)
                else:
                    # for GRUs, states are single tensors
                    hidden = hidden.index_select(1, select_indices)

            if att_vectors is not None:
                att_vectors = att_vectors.index_select(0, select_indices)

        def pad_and_stack_hyps(hyps, pad_value):
            filled = (
                np.ones((len(hyps), max([h.shape[0] for h in hyps])), dtype=int) * pad_value
            )
            for j, h in enumerate(hyps):
                for k, i in enumerate(h):
                    filled[j, k] = i
            return filled

        # from results to stacked outputs
        assert n_best == 1
        # only works for n_best=1 for now
        final_outputs = pad_and_stack_hyps(
            [r[0].cpu().numpy() for r in results["predictions"]], pad_value=pad_index
        )

        return final_outputs, None
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
        res_dict = self.sign_encoder.forward(x, len_x_all)
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
            # "framewise" : res_dict['framewise'],
            "sign_feat_teacher": bert_output_classifier,
            "sequence_logits_teacher": sequence_logits,
            # "sequence_logits_sign" : sequence_logits_sign,
            # "recognized_sents": pred,
            # "recognized_sents_sign": pred_sign,
            # "next_sentence_loss": next_sentence_loss
        }
    def forward_train_gloss2text(self, label_word_with_SOS, label_word_with_SOS_lgt, gloss, gloss_lgt):
        # 输入不包括<eos>
        batch = label_word_with_SOS.size(0)
        unroll_steps = label_word_with_SOS.size(1)
        trg_embed = self.text_embedding(vis_feats=None,vis_pe=None,input_ids=label_word_with_SOS)

        mask_text = torch.zeros(batch,max(gloss_lgt), max(gloss_lgt) , dtype=torch.long,device=gloss_lgt.device)
        for i in range(batch):
            mask_text[i] = self.get_attn_mask(gloss_lgt[i],max(gloss_lgt))

        trg_mask = torch.zeros(batch,max(label_word_with_SOS_lgt), max(label_word_with_SOS_lgt) , dtype=torch.bool,device=label_word_with_SOS_lgt.device)
        for i in range(batch):
            trg_mask[i] = self.get_attn_mask_boolean(label_word_with_SOS_lgt[i],max(label_word_with_SOS_lgt))

        src_mask = torch.zeros(batch,max(gloss_lgt), max(gloss_lgt) , dtype=torch.bool,device=gloss_lgt.device)
        for i in range(batch):
            src_mask[i] = self.get_attn_mask_boolean(gloss_lgt[i],max(gloss_lgt))
        
        # src_mask = trg_mask
        encoder_output = self.bert_gloss2text(vis_feats=None,vis_pe=None,input_ids=gloss,attention_mask=mask_text)
        decoder_outputs = self.decoder_gloss2text(
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

    def forward_train_student(self, x, len_x_all,label_word, label_word_lgt):
        batch, temp, channel, height, width = x.shape
        res_dict = self.sign_encoder_2(x, len_x_all)
        lgt = res_dict['feat_len']
        conv_logits = res_dict['conv_logits']
        sequence_logits_sign = res_dict['sequence_logits_sign']
        sign_feat = res_dict['sign_feat']
        sign_feat_student = self.signToText(res_dict['sign_feat'])

        pred_sign,peak_timestep = self.decoder_combine.decode(sequence_logits_sign, lgt, batch_first=False, probs=False,search_mode='peak')
        # sequence_logits_teacher = sequence_logits_teacher
        # print(pred_sign)
        unroll_steps = label_word.size(1)
        trg_embed = self.text_embedding(vis_feats=None,vis_pe=None,input_ids=label_word)
        trg_mask = torch.zeros(batch,max(label_word_lgt), max(label_word_lgt) , dtype=torch.bool,device=label_word.device)
        for i in range(batch):
            trg_mask[i] = self.get_attn_mask_boolean(label_word_lgt[i],max(label_word_lgt))
        
        src_mask = torch.zeros(batch,max(lgt), max(lgt) , dtype=torch.bool,device=sequence_logits_sign.device)
        for i in range(batch):
            src_mask[i] = self.get_attn_mask_boolean(lgt[i],max(lgt)) 
        decoder_outputs = self.decoder_sign2text(
            trg_embed=trg_embed, 
            encoder_output=sign_feat.transpose(0,1),
            src_mask=src_mask[:,0:1,:],
            trg_mask=trg_mask,        
                # hidden=decoder_hidden,
        )
        word_outputs, word_feature, _, _ = decoder_outputs
        # Calculate Translation Loss
        txt_log_probs = F.log_softmax(word_outputs, dim=-1)


        return {
            "feat_len": lgt,
            "conv_logits" : conv_logits,
            "sign_feat_student": sign_feat_student,
            "sequence_logits_sign" : sequence_logits_sign,
            "peak_timestep" : peak_timestep,
            "word_outputs": word_outputs,
            "txt_log_probs": txt_log_probs,
            "word_feature": word_feature

        }

    def forward_test(self, x, len_x_all,label_word, label_word_lgt, translation_beam_size, translation_beam_alpha):
        batch, temp, channel, height, width = x.shape
        # batch = x.size(0)
        # res_dict_teacher = self.sign_encoder.forward_framewiseOnly(x,len_x_all)
        res_dict = self.sign_encoder_2(x, len_x_all)
        lgt = res_dict['feat_len']
        conv_logits = res_dict['conv_logits']
        sequence_logits_sign = res_dict['sequence_logits_sign']
        sign_feat = res_dict['sign_feat']

        pred_sign = self.decoder_combine.decode(sequence_logits_sign, lgt, batch_first=False, probs=False)
        
        unroll_steps = label_word.size(1)
        trg_embed = self.text_embedding(vis_feats=None,vis_pe=None,input_ids=label_word)
        trg_mask = torch.zeros( (batch,max(label_word_lgt), max(label_word_lgt)) , dtype=torch.bool,device=label_word.device)
        for i in range(batch):
            trg_mask[i] = self.get_attn_mask_boolean(label_word_lgt[i],max(label_word_lgt))
        
        src_mask = torch.zeros(batch,max(lgt), max(lgt) , dtype=torch.bool,device=sequence_logits_sign.device)
        for i in range(batch):
            src_mask[i] = self.get_attn_mask_boolean(lgt[i],max(lgt)) 
        if translation_beam_size < 2:
            stacked_txt_output, stacked_attention_scores = greedy(
                encoder_hidden = None,
                encoder_output = sign_feat.transpose(0,1),
                src_mask = src_mask[:,0:1,:],
                embed = self.text_embedding,
                bos_index = 1,
                eos_index = 2,
                decoder=self.decoder_sign2text,
                max_output_length = 30,
            )
                # batch, time, max_sgn_length
        else:  # beam size
            stacked_txt_output, stacked_attention_scores = beam_search(
                size=translation_beam_size,
                encoder_hidden = None,
                encoder_output = sign_feat.transpose(0,1),
                src_mask = src_mask[:,0:1,:],
                embed = self.text_embedding,
                max_output_length = 30,
                alpha=translation_beam_alpha,
                eos_index=2,
                pad_index=0,
                bos_index=1,
                decoder=self.decoder_sign2text,
            )

        return {
            "feat_len": lgt,
            "sequence_logits_sign" : sequence_logits_sign,
            "recognized_sents": pred_sign,
            'stacked_txt_output': stacked_txt_output,
            'stacked_attention_scores': stacked_attention_scores
        }

    def criterion_calculation(self, ret_dict_teacher,ret_dict_teacher_text2gloss, ret_dict_student,label, label_lgt, label_text_target,epoch,loss_weights):
        loss = 0
        
        # E_label_lgt=torch.ones(4)
        loss_dist_fusion_to_slr = torch.tensor(0).cuda().float()
        loss_dist_fusion_to_slr_feature = torch.tensor(0).cuda().float()
        loss_ctc_RNN = torch.tensor(0).cuda().float()
        loss_ctc_conv1d = torch.tensor(0).cuda().float()
        loss_dist_RNN_to_conv1d = torch.tensor(0).cuda().float()
        loss_sign_translation = torch.tensor(0).cuda().float()
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
            elif k == 'DistFusion2SLR':
                loss_dist_fusion_to_slr = self.loss['distillation'](ret_dict_student["sequence_logits_sign"],
                                                           ret_dict_teacher["sequence_logits_teacher"].detach(),
                                                           use_blank=False)
                loss += weight * loss_dist_fusion_to_slr
            
            elif k == 'DistFusion2SLRFeature':
                loss_dist_fusion_to_slr_feature = self.loss['distillation_feature'](ret_dict_student["sign_feat_student"],
                                                           ret_dict_teacher["sign_feat_teacher"].detach(),
                                                           use_blank=False)
                loss += weight * loss_dist_fusion_to_slr

            elif k == 'DistTranslation2SLR':
                
                loss_dist_translation_to_slr += self.loss['distillation'](ret_dict_student['word_outputs'].transpose(0,1),
                                                                ret_dict_teacher_text2gloss["word_outputs"].transpose(0,1).detach(),
                                                                use_blank=False)
                
                loss += weight * loss_dist_translation_to_slr
            
            elif k == 'DistTranslation2SLRFeature':
                
                loss_dist_translation_to_slr_feature += self.loss['distillation_feature'](ret_dict_student['word_feature'].transpose(0,1),
                                                                ret_dict_teacher_text2gloss["word_feature"].transpose(0,1).detach()
                                                                )
                
                loss += weight * loss_dist_translation_to_slr_feature

            #     loss += weight * loss_dist_fusion_to_slr
            
            elif k == 'TranslationGlossCrossEntropy':
                loss += weight * self.loss['translation'](
                    ret_dict_teacher_text2gloss['txt_log_probs'], label_text_target,
                )
            elif k == 'TranslationCrossEntropy':
                loss_sign_translation = self.loss['translation'](
                    ret_dict_student['txt_log_probs'], label_text_target,
                )
                loss += weight * loss_sign_translation
            
            

            
            
            

        return loss,loss_ctc_RNN,loss_ctc_conv1d,loss_dist_RNN_to_conv1d,loss_dist_fusion_to_slr,loss_dist_translation_to_slr,loss_sign_translation

    def criterion_init(self):
        self.loss['CTCLoss'] = torch.nn.CTCLoss(reduction='none', zero_infinity=False)
        self.loss['distillation'] = SeqKD(T=8)
        self.loss['translation'] = XentLoss(
            pad_index=0, smoothing=0.0
        )
        self.loss['distillation_feature'] = torch.nn.MSELoss(reduction='mean')
        # self.loss['distillation_disentangle'] = SeqKD(T=8)
        return self.loss
