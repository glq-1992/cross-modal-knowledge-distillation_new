import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math



def create_src_lengths_mask(batch_size, src_lengths):
    '''
    生成布尔掩码以防止注意力超出source的末尾
    :param batch_size: int
    :param src_lengths: [batch_size] 每个句子的实际长度
    :return: [batch_size, max_src_len]
    '''
    max_src_len = src_lengths.max()
    # [1, max_src_len]
    src_indices = torch.arange(0, max_src_len).unsqueeze(0).type_as(src_lengths)
    # [batch_size, max_src_len]
    src_indices = src_indices.expand(batch_size, max_src_len)
    src_lengths = src_lengths.unsqueeze(1).expand(batch_size, max_src_len)
    # 小于实际长度的为1，大于的为0，detach截断反向梯度传播
    return (src_indices < src_lengths).int().detach()
 
 
def masked_softmax(scores, src_lengths, src_length_masking=True):
    '''
    先生成mask,然后再进行softmax。
    '''
    if src_length_masking:
        batch_size, max_src_len = scores.size()
        # compute masks
        src_mask = create_src_lengths_mask(batch_size, src_lengths)
        # Fill pad positions with -inf
        scores = scores.masked_fill(src_mask == 0, -np.inf)
 
    # 转换为float16，然后再次转换回来以防止loss爆炸
    return F.softmax(scores.float(), dim=-1).type_as(scores)


class BaseAttention(nn.Module):
    def __init__(self, decoder_hidden_state_dim, context_dim):
        super(BaseAttention, self).__init__()
        self.decoder_hidden_state_dim = decoder_hidden_state_dim
        self.context_dim = context_dim
 
    def forward(self, decoder_state, src_hids, src_lengths):
        '''
        :param decoder_state: bsz * decoder_hidden_state_dim
        :param src_hids: src_len * bsz * context_dim
        :param src_lengths: bsz * 1, actual sequence lens
        :return:
        outputs: bsz * context_dim
        attn_scores: max_src_len * bsz
        '''
        raise NotImplementedError

class MLPAttention(BaseAttention):
    '''
     alpha_ij = V_a * tanh(W_ae * enc_i + W_ad * dec_j + b_a)
    '''
    def __init__(self, decoder_hidden_state_dim, context_dim, **kwargs):
        super().__init__(decoder_hidden_state_dim, context_dim)
 
        self.context_dim = context_dim
        self.attention_dim = kwargs.get('attention_dim', context_dim)
        # W_ae and b_a
        self.encoder_proj = nn.Linear(context_dim, self.attention_dim, bias=True)
        # W_ad
        self.decoder_proj = nn.Linear(decoder_hidden_state_dim, self.attention_dim)
        # V_a
        self.to_scores = nn.Linear(self.attention_dim, 1, bias=False)
        self.src_length_masking = kwargs.get('src_length_masking', True)
 
    def prepare_for_onnx_export_(self, **kwargs):
        self.src_length_masking = False
 
    def forward(self, decoder_state, source_hids, src_lengths):
        """The expected input dimensions are:
        decoder_state: bsz x decoder_hidden_state_dim
        source_hids: src_len x bsz x context_dim
        src_lengths: bsz
        """
        src_len, bsz, _ = source_hids.size()
        # (src_len*bsz) x context_dim (to feed through linear)
        flat_source_hids = source_hids.contiguous().view(-1, self.context_dim)
        # (src_len*bsz) x attention_dim
        encoder_component = self.encoder_proj(flat_source_hids)
        # src_len x bsz x attention_dim
        encoder_component = encoder_component.contiguous().view(src_len, bsz, self.attention_dim)
        # 1 x bsz x attention_dim
        decoder_component = self.decoder_proj(decoder_state).unsqueeze(0)
        # Sum with broadcasting and apply the non linearity
        # src_len x bsz x attention_dim
        hidden_att = F.tanh(
            (decoder_component + encoder_component).contiguous().view(-1, self.attention_dim)
        )
        # Project onto the reals to get attentions scores (bsz x src_len)
        attn_scores = self.to_scores(hidden_att).contiguous().view(src_len, bsz).t()
 
        # Mask + softmax (src_len x bsz)
        normalized_masked_attn_scores = masked_softmax(
            attn_scores, src_lengths, self.src_length_masking
        ).t()
 
        # Sum weighted sources (bsz x context_dim)
        attn_weighted_context = (
                source_hids * normalized_masked_attn_scores.unsqueeze(2)
        ).sum(0)
 
        return attn_weighted_context, normalized_masked_attn_scores


def Linear(in_features, out_features, bias=True):
    """Weight-normalized Linear layer (input: N x T x C)"""
    m = nn.Linear(in_features, out_features, bias=bias)
    m.weight.data.uniform_(-0.1, 0.1)
    if bias:
        m.bias.data.uniform_(-0.1, 0.1)
    return m
 
 
# class DotAttention(BaseAttention):
#     def __init__(self, decoder_hidden_state_dim, context_dim, **kwargs):
#         super().__init__(decoder_hidden_state_dim, context_dim)
 
#         self.input_proj = None
#         force_projection = kwargs.get('force_projection', False)
#         if force_projection or decoder_hidden_state_dim != context_dim:
#             self.input_proj = Linear(decoder_hidden_state_dim, context_dim, bias=True)
#         self.src_length_masking = kwargs.get('src_length_masking', True)
 
#     def prepare_for_onnx_export(self, **kwargs):
#         self.src_length_masking = False
 
#     def forward(self, decode_state, source_hids, src_lengths):
#         '''
#         :param decoder_state: bsz * decoder_hidden_state_dim
#         :param src_hids: src_len * bsz * context_dim
#         :param src_lengths: bsz * 1, actual sequence lens
#         :return:
#         outputs: bsz * context_dim
#         attn_scores: max_src_len * bsz
#         '''
#         # Reshape to bsz x src_len x context_dim
#         source_hids = source_hids.transpose(0, 1)
#         # decode_state: [bsz, context_dim]
#         if self.input_proj is not None:
#             decode_state = self.input_proj(decode_state)
#         # compute attention [bsz, src_len, context_dim] * [bsz, context_dim, 1]
#         attn_score = torch.bmm(source_hids, decode_state.unsqueeze(2)).squeeze(2)
 
#         # Mask + softmax(bsz * src_lens)
#         normalized_masked_attn_scores = attention_utils.masked_softmax(
#             attn_score, src_lengths, self.src_length_masking)
 
#         # Sum weighted sources
#         attn_weighted_context = ((source_hids * normalized_masked_attn_scores.unsqueeze(2)) \
#                                  .contiguous().sum(1))
#         return attn_weighted_context, normalized_masked_attn_scores.t()






def sequence_mask(X, valid_len, value=0):
    """Mask irrelevant entries in sequences."""
    maxlen = X.size(1)
    # 返回batchsize*max_length的tensor mask, mask[i]代表第i个序列的每一步是否应该被-1e6替换
    mask = torch.arange((maxlen), dtype=torch.float32,
                        device=X.device)[None, :] < valid_len[:, None]
    X[~mask] = value
    return X

def masked_softmax(X, valid_lens):
    """通过在最后一个轴上遮蔽元素来执行 softmax 操作"""
    # `X`: 3D张量, `valid_lens`: 1D或2D 张量
    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        # 在最后的轴上，被遮蔽的元素使用一个非常大的负值替换，从而其 softmax (指数)输出为 0
        X =sequence_mask(X.reshape(-1, shape[-1]), valid_lens, value=-1e6)
        return nn.functional.softmax(X.reshape(shape), dim=-1)

class DotProductAttention(nn.Module):
    """Scaled dot product attention."""
    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    # Shape of `queries`: (`batch_size`, no. of queries, `d`)
    # Shape of `keys`: (`batch_size`, no. of key-value pairs, `d`)
    # Shape of `values`: (`batch_size`, no. of key-value pairs, value dimension)
    # Shape of `valid_lens`: (`batch_size`,) or (`batch_size`, no. of queries)
    def forward(self, queries, keys, values, valid_lens=None):
        d = queries.shape[-1]
        # Set `transpose_b=True` to swap the last two dimensions of `keys`
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)


class LinearDotProductAttention(nn.Module):
    """Scaled dot product attention."""
    def __init__(self, dropout, in_features,out_features,**kwargs):
        super(LinearDotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear_Q = nn.Linear(in_features, out_features)
        self.linear_K = nn.Linear(in_features, out_features)
        self.linear_V = nn.Linear(in_features, out_features)

    # Shape of `queries`: (`batch_size`, no. of queries, `d`)
    # Shape of `keys`: (`batch_size`, no. of key-value pairs, `d`)
    # Shape of `values`: (`batch_size`, no. of key-value pairs, value dimension)
    # Shape of `valid_lens`: (`batch_size`,) or (`batch_size`, no. of queries)
    def forward(self, queries, keys, values, valid_lens=None):
        d = queries.shape[-1]
        queries=self.linear_Q(queries)
        keys=self.linear_K(keys)
        values=self.linear_V(values)
        # Set `transpose_b=True` to swap the last two dimensions of `keys`
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)

# attention = DotProductAttention(dropout=0.5)

# batch_size = 2
# x_len , y_len = 3, 8
# in_dim, out_dim = 2, 10

# Q = torch.ones((batch_size, x_len, in_dim))
# K = torch.ones((batch_size, y_len, in_dim))
# V = torch.ones((batch_size, y_len, out_dim))

# soft_attention_ans = attention(Q, K, V)
# self_attention_ans = attention(Q, Q, Q)
# print(soft_attention_ans.shape)
# print(self_attention_ans.shape)
