# 基于多模态Transformer的对话场景手语识别和手语翻译&&基于多模态知识蒸馏的手语识别与翻译
包含史鹏的学位论文《面向对话场景的多模态手语识别与翻译方法研究》中两项工作的代码

基于Visual Alignment Constraint for Continuous Sign Language Recognition（VAC）实现，README_vac.md是VAC原版的说明文档

## 运行环境
---
使用该代码进行实验时的运行环境可使用Conda从requirements.yml复原，其中包含一些不需要的依赖。
最关键的几项依赖见README_vac.md


---
## 结构说明
main_xxx.py 主程序

    if __name__ == '__main__': 读取设置、加载模型和数据集、进行训练或测试
    Processor：初始化方法中加载模型和数据集，start方法根据设置中的phase为train、dev、test，进行训练或测试

seq_scripts_xxx.py 包含训练和测试的流程 

    seq_train 训练
    seq_test 测试
    
utils/: 各项工具

    decode_xxx.py CTC解码器的不同实现
    device.py 将模型加载到GPU
    optimizer.py 优化器
    parameters_xxx.py 使用argparse实现的，训练或测试的设置，其中--config项的值为yaml文件的路径，可修改该yaml文件以修改设置

slr_xxx.py 手语识别模型 slr_slt_xxx.py 手语识别与翻译的模型

    SLRModel类：手语识别模型or手语识别与翻译模型，forward_train和forward_test方法分别用于训练和测试
    criterion_calculation：损失函数计算
    criterion_init：损失函数初始化

software/: sclite软件的软连接，用于评估手语识别的精度。

signjoey/: Transformer的实现，只有decoder被用到

pytorch_pretrained_bert/: BERT的实现

modules/: 部分模型结构的实现

models/: 部分模型结构的实现，没有被用到

mmseg/: 部分模型结构的实现，没有被用到

evaluation/: 用于手语识别评估，每一个slr_eval_xxx/ 子目录下，是用于评估手语识别模型在该数据集精度的文件

DatasetFile/: 每个子目录包含了一个数据集的相关标注

dataset/: dataloader_xxx.py 重写torch.utils.data.dataset类，用于读取数据

ctc/: 一个ctc解码的实现，没用被用到

configs/：
    
    \_base\_/ 和 swin/ 无用
    baseline_xxx.yaml 设置文件，包含训练和测试中的各种设置
    数据集名称_yaml 数据集文件，dataset_root和dict_path无用，evaluation_dir和evaluation_prefix记录了该使用evaluation/目录下的那个文件，进行手语识别的评估


---
## 其他
本文档只对仓库内代码进行说明，具体两个工作如何训练和测试，见以下文档：

1.基于多模态Transformer的对话场景手语识别：

    README_dialogue.md

2.基于多模态知识蒸馏的手语识别与翻译：

    README_distill.md

1为2提供了教师模型，因此在进行2的训练时，需要先训练好1
