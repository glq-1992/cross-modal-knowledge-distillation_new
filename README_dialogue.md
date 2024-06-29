# 基于多模态Transformer的对话场景手语识别与翻译

## 训练与测试
手语识别训练：

    python main_pheonix_bert_sign_text_twostream_hybridattnfusion_res_nsp_difflr.py
    对应的config文件为baseline_pheonix_bert_sign_text_twostream_hybridattnfusion_res_nsp.yaml


手语识别测试：

    python main_pheonix_bert_sign_text_twostream_hybridattnfusion_res_nsp_difflr.py --load-weights 保存的权重路径 --phase test

手语翻译训练：

    python main_pheonix_bert_sign_text_twostream_hybridattnfusion_res_nsp_difflr_slt.py
    对应的config文件为baseline_slt_pheonix_bert_sign_text_twostream_hybridattnfusion_res_nsp.yaml，可以通过main文件追溯到

手语翻译测试：

    python main_pheonix_bert_sign_text_twostream_hybridattnfusion_res_nsp_difflr_slt.py --load-weights 保存的权重路径 --phase test

如何更改训练or测试参数：使用命令行参数，或更改config文件，否则为argparse中的默认值

## 模型说明

- 文件名中的hybridattnfusion：论文中的多模态Transformer encoder
- 文件名中的selfattnfusion：普通Transformer encoder
- 文件名中有nsp的，都可以做文本-手语是否匹配的loss（相当于BERT中的Next Sentence Predict），但这个loss在实验中总会降低精度，所以论文中不使用
- 文件名中有mlm的，都可以做Mask language modeling，但这个loss在实验中总会降低精度，所以论文中不使用


- slr_slt_bert_network_sign_text_twostream_hybridattnfusion_res_GSL_nsp_vac_pheonix_loadPretrain_2inputdecoder.py
    
    论文中的最终版本模型，使用了多模态Transformer encoder和双注意力decoder，同时进行手语识别与翻译任务，

- slr_slt_bert_network_sign_text_twostream_hybridattnfusion_res_GSL_nsp_vac_pheonix_loadPretrain.py

    多模态Transformer encoder+普通decoder，普通decoder的输入为多模态Transformer的输出中，手语视频部分的特征

- slr_slt_bert_network_sign_text_twostream_hybridattnfusion_res_GSL_nsp_vac_pheonix_loadPretrain_useAllOutput.py

    多模态Transformer encoder+普通decoder，普通decoder的输入为多模态Transformer的全部输出

- slr_slt_bert_network_sign_text_twostream_selfattnfusion_res_GSL_nsp_vac_pheonix_loadPretrain.py

    手语视频与文本特征拼接后，送入普通Transformer encoder+普通decoder

- 以上模型去除"slt"即为只做手语识别不做手语翻译的模型

## 一些尝试过的模型
- slr_bert_network_sign_text_twostream_hybridattnfusion_res_GSL_nsp_1dcnn_pheonix_loadPretrain.py

    手语编码器中不使用RNN

- slr_bert_network_sign_text_twostream_hybridattnfusion_res_GSL_nsp_vac_pheonix_loadPretrain_mlm.py

    在手语识别任务的基础上增加mask language modeling自监督任务,mlm的分类器和text encoder的embedding共享参数

- slr_bert_network_sign_text_twostream_hybridattnfusion_res_GSL_nsp_vac_pheonix_loadPretrain_mlm_nosharemlm.py

    mlm的分类器不和text encoder的embedding共享参数

- slr_bert_network_sign_text_twostream_selfattnfusion_res_GSL_nsp_localtransformer.py

    局部Transformer（处理大小为3的滑动窗口）+全局Transformer

- slr_bert_network_sign_text_twostream_selfattnfusion_res_GSL_nsp_localtransformer.py

    局部Transformer（处理大小为3的滑动窗口）+全局Transformer

- slr_bert_network_sign_text_twostream_selfattnfusion_res_GSL_nsp_localtransformer_multibranch.py

    局部Transformer（处理大小为3的滑动窗口）+全局Transformer，同时输入左手、右手（使用hrnet提取）和视频帧，融合三种特征

- slr_bert_network_sign_text_twostream_selfattnfusion_res_GSL_nsp_vac_pheonix_loadPretrain_channelmask_xxx.py

    在视频特征的通道维度做mask，然后经过Transformer再将其复原，相当于视觉-文本预训练中的masked image modeling，没用





