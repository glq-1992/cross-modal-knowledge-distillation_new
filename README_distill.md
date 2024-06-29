# 基于多模态知识蒸馏的手语识别与翻译

## 训练与测试
手语识别训练：

    python main_pheonix_bert_sign_text_twostream_hybridattnfusion_res_nsp_difflr_distillation_slrstudent_2teacher.py
    对应的config文件为baseline_pheonix_bert_sign_text_twostream_hybridattnfusion_res_nsp_distillation_slrstudent_2teacher.yaml

手语识别测试：

    python main_pheonix_bert_sign_text_twostream_hybridattnfusion_res_nsp_difflr_distillation_slrstudent_2teacher.py --load-weights 保存的权重路径 --phase test

手语翻译训练：

    python main_pheonix_bert_sign_text_twostream_hybridattnfusion_res_nsp_difflr_distillation_slrstudent_2teacher_slt_loadmultimodalslt.py
    对应的config文件为baseline_slt_pheonix_bert_sign_text_twostream_hybridattnfusion_res_nsp_distillation_slrstudent_2teacher.yaml

手语翻译测试：

    python main_pheonix_bert_sign_text_twostream_hybridattnfusion_res_nsp_difflr_distillation_slrstudent_2teacher_slt_loadmultimodalslt.py --load-weights 保存的权重路径 --phase test

如何更改训练or测试参数：使用命令行参数，或更改config文件，否则为argparse中的默认值

## 模型说明


- 1.slr_bert_network_sign_text_twostream_hybridattnfusion_res_GSL_nsp_vac_pheonix_loadPretrain_slrstudent_2teacher_feature_new.py
    
    使用对话模型和text2gloss模型知识蒸馏slr模型


-  2.slr_bert_network_sign_text_twostream_hybridattnfusion_res_GSL_nsp_vac_pheonix_loadPretrain_transformerstudent_2teacher_feature_new

    把学生模型的bilstm换成transformer，其他和1一样

-  3.slr_slt_bert_network_sign_text_twostream_hybridattnfusion_res_GSL_nsp_vac_pheonix_loadPretrain_slrstudent_2teacher_feature

    使用对话模型和gloss2text模型知识蒸馏slt模型

-  4.slr_slt_bert_network_sign_text_twostream_hybridattnfusion_res_GSL_nsp_vac_pheonix_loadPretrain_transformerstudent_2teacher_feature

    把学生模型的bilstm换成transformer，其他和3一样

- 其他所有文件名带“student”的知识蒸馏模型，都是以上4个的弱化版（减少了部分蒸馏损失函数），可以通过以上4个模型+不同的损失函数设置得到


## 1.手语识别训练注意事项

从头训练：由于是知识蒸馏方法，需要先训练好教师模型，然后读取教师模型。

    步骤1：训练对话场景手语识别模型（第一个教师模型）。用README_dialogue.md中的方法训练slr_bert_network_sign_text_twostream_hybridattnfusion_res_GSL_nsp_vac_pheonix_loadPretrain_2inputdecoder.py 中的对话场景手语识别

    步骤2：在load_weights里填入步骤1中保存的模型权重路径，运行python main_pheonix_bert_sign_text_twostream_hybridattnfusion_res_nsp_difflr_distillation_slrstudent_2teacher.py

    从头训练时，应视情况调整optimizer_args中的base_lr和step，由于在第30个epoch才会开始训练学生模型，step（在第几个epoch学习率减半）应设的大一些，建议为[ 50, 70,90]


目前的baseline_pheonix_bert_sign_text_twostream_hybridattnfusion_res_nsp_distillation_slrstudent_2teacher.yaml中没有使用load_weights，而是使用了load_checkpoints。load_checkpoints的路径是已经训练了30个epoch的权重，即已经在步骤2训练了30个epoch。

## 2.手语翻译训练注意事项
从头训练：

    步骤1：训练对话场景手语翻译模型（第一个教师模型）。用README_dialogue.md中的方法训练slr_slt_bert_network_sign_text_twostream_hybridattnfusion_res_GSL_nsp_vac_pheonix_loadPretrain_2inputdecoder.py 中的对话场景手语翻译模型

    步骤2：在load_weights_multimodalteacher里填入填入步骤1中保存的模型权重路径，（可选）在load_weights_encoder2里填入使用知识蒸馏方法训练出的手语识别模型的权重路径（论文中有提及，手语翻译会读取手语识别的权重作为编码器权重）。python main_pheonix_bert_sign_text_twostream_hybridattnfusion_res_nsp_difflr_distillation_slrstudent_2teacher_slt_loadmultimodalslt.py 进行训练。

## 代码中损失函数与论文的关系
首先ConvCTCSign  SeqCTCSign  Dist 是手语识别硬标签损失函数（见VAC） ，TranslationCrossEntropy 是手语翻译硬标签损失函数，保持不变即可

手语识别：

    TranslationToGlossCrossEntropy是训练Text2Gloss模型的损失函数，设为1即可，在seq文件中会根据epoch变换。
    DistFusion2SLR 对应于论文公式(4-13) beta
    FeatureFusion2SLR  对应于论文公式(4-13) gamma
    DistTranslation2SLR 对应于论文公式(4-13) delta
    FeatureTranslation2SLR 对应于论文公式(4-13) epsilon

手语翻译：

    TranslationToGlossCrossEntropy是训练Text2Gloss模型的损失函数，设为1即可，在seq文件中会根据epoch变换。
    DistFusion2SLR 对话场景教师模型的概率分布损失函数，应设为0， （加入该知识蒸馏损失会降低精度）
    DistFusion2SLRFeature 对话场景教师模型的特征损失函数，应设为0，
    DistTranslation2SLR 对应于论文公式(4-19) gamma
    DistTranslation2SLRFeature 对应于论文公式(4-19) delta
        





