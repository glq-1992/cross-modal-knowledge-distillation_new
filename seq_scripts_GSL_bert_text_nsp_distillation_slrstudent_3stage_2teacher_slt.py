import os
import pdb
import sys
import copy
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt
from evaluation.slr_eval.wer_calculation import evaluate
from utils.wer_calculate import get_wer_delsubins

from signjoey.metrics import bleu, chrf, rouge, wer_list

PAD_TOKEN = "<pad>"
BOS_TOKEN = "<s>"
EOS_TOKEN = "</s>"

def seq_train(loader, model, optimizer, device, loss_weights_original,epoch_idx, recoder):
    # model=device.model_to_device(model)
    
    model.train()
    loss_value = []
    loss_ctc_RNN_value = []
    loss_ctc_conv1d_value = []
    loss_dist_RNN_to_conv1d_value = []
    loss_dist_fusion_to_slr_value = []
    loss_dist_translation_to_slr_value = []
    loss_sign_translation_value = []
    loss_weights = copy.deepcopy(loss_weights_original)
    # 前50个epoch只训练gloss2sign
    if epoch_idx < 50:
        loss_weights.pop('DistTranslation2SLR')
        loss_weights.pop('ConvCTCSign')
        loss_weights.pop('SeqCTCSign')
        loss_weights.pop('Dist')
        loss_weights.pop('DistFusion2SLR')
        loss_weights.pop('TranslationCrossEntropy')
    
    else:
        loss_weights.pop('TranslationToGlossCrossEntropy')
    clr = [group['lr'] for group in optimizer.optimizer.param_groups]
    for batch_idx, data in enumerate(loader):
        vid = device.data_to_device(data[0])
        vid_lgt = device.data_to_device(data[1])
        label = device.data_to_device(data[2])
        label_lgt = device.data_to_device(data[3])
        gloss = device.data_to_device(data[4])
        label_text = device.data_to_device(data[5])
        label_text_lgt = device.data_to_device(data[6])
        label_text_target = device.data_to_device(data[7])
        text = device.data_to_device(data[8])
        text_lgt = device.data_to_device(data[9])
        next_sentence_label = device.data_to_device(data[10])
        if epoch_idx < 50:
            ret_dict_teacher_translation = model.forward_train_gloss2text(label_text, label_text_lgt, gloss, label_lgt)
            loss,loss_ctc_RNN,loss_ctc_conv1d,loss_dist_RNN_to_conv1d,loss_dist_fusion_to_slr,loss_dist_translation_to_slr,loss_sign_translation = model.criterion_calculation(0, ret_dict_teacher_translation, 0,label, label_lgt, label_text_target,epoch_idx,loss_weights)
        else:
            ret_dict_teacher_context = None
            ret_dict_teacher_translation = None
            with torch.no_grad():
                ret_dict_teacher_context = model.forward_train(vid, vid_lgt,label, label_lgt, text, text_lgt,next_sentence_label)
            with torch.no_grad():
                ret_dict_teacher_translation = model.forward_train_gloss2text(label_text, label_text_lgt, gloss, label_lgt)
            ret_dict_student = model.forward_train_student(vid, vid_lgt,label_text, label_text_lgt)
            loss,loss_ctc_RNN,loss_ctc_conv1d,loss_dist_RNN_to_conv1d,loss_dist_fusion_to_slr,loss_dist_translation_to_slr,loss_sign_translation = model.criterion_calculation(ret_dict_teacher_context, ret_dict_teacher_translation, ret_dict_student,label, label_lgt, label_text_target,epoch_idx,loss_weights)

        if np.isinf(loss.item()) or np.isnan(loss.item()):
            print(data[-1])
            print('loss is nan or inf!')

            continue
        
        optimizer.zero_grad()
        loss.backward()
        # nn.utils.clip_grad_norm_(model.rnn.parameters(), 5)
        optimizer.step()
        loss_value.append(loss.item())
        loss_ctc_RNN_value.append(loss_ctc_RNN.item())
        loss_ctc_conv1d_value.append(loss_ctc_conv1d.item())
        loss_dist_RNN_to_conv1d_value.append(loss_dist_RNN_to_conv1d.item())
        loss_dist_fusion_to_slr_value.append(loss_dist_fusion_to_slr.item())
        loss_dist_translation_to_slr_value.append(loss_dist_translation_to_slr.item())
        loss_sign_translation_value.append(loss_sign_translation.item())

        if batch_idx % recoder.log_interval == 0:
            recoder.print_log(
                '\tEpoch: {}, Batch({}/{}) done. Loss: {:.8f}  lr:{:.8f}'
                    .format(epoch_idx, batch_idx, len(loader), loss.item(), clr[0]))
    optimizer.scheduler.step()
    recoder.print_log('\tMean training loss: {:.10f} loss_ctc_RNN: {:.10f} loss_ctc_conv1d: {:.10f} loss_dist_RNN_to_conv1d: {:.10f} loss_dist_fusion_to_slr: {:.10f} loss_dist_translation_to_slr: {:.10f} loss_translation: {:.10f}.'.format(np.mean(loss_value),np.mean(loss_ctc_RNN_value),np.mean(loss_ctc_conv1d_value),np.mean(loss_dist_RNN_to_conv1d_value),np.mean(loss_dist_fusion_to_slr_value),np.mean(loss_dist_translation_to_slr_value),np.mean(loss_sign_translation_value)))
    # exit()
    # return loss_value



def seq_eval(cfg, loader, model, device, mode, epoch, work_dir, recoder,
             evaluate_tool="python"):
    translation_beam_size = 1
    translation_beam_alpha = 0
    model.eval()
    total_sent = []
    total_info = []
    total_conv_sent = []

    # 翻译结果
    all_txt_outputs = []
    all_attention_scores = []
    ref_text = []
    # stat = {i: [0, 0] for i in range(len(loader.dataset.dict))}
    sample_count = 0
    wer_sum = np.zeros([4])
    for batch_idx, data in enumerate(tqdm(loader)):
        recoder.record_timer("device")
        vid = device.data_to_device(data[0])
        vid_lgt = device.data_to_device(data[1])
        label = device.data_to_device(data[2])
        label_lgt = device.data_to_device(data[3])
        gloss = device.data_to_device(data[4])
        label_text = device.data_to_device(data[5])
        label_text_lgt = device.data_to_device(data[6])
        label_text_target = device.data_to_device(data[7])
        text = device.data_to_device(data[8])
        text_lgt = device.data_to_device(data[9])
        next_sentence_label = device.data_to_device(data[10])
        # info = data[6]
        
        with torch.no_grad():
            ret_dict = model.forward_test(vid, vid_lgt,label_text,label_text_lgt,translation_beam_size,translation_beam_alpha)
        
        total_info += [file_name for file_name in data[-1]]
        total_sent += ret_dict['recognized_sents']

        all_txt_outputs.extend(ret_dict['stacked_txt_output'])
        all_attention_scores.extend(
                ret_dict['stacked_attention_scores']
                if ret_dict['stacked_attention_scores'] is not None
                else []
        )
        ref_text.extend(label_text_target.cpu().numpy())

        # total_conv_sent += ret_dict['conv_sents']
        # start = 0
        # for i,label_length in enumerate(label_lgt):
        #     end = start + label_length
        #     ref = label[start:end].tolist()
        #     hyp = [x for x in ret_dict['recognized_sents_notext'][i]] 
        #     wer = get_wer_delsubins(ref, hyp)
        #     wer_sum += wer
        #     sample_count += 1
        #     start = end

            # print(wer)

    # print('wer:' + str(wer_sum / sample_count))
        # total_conv_sent += ret_dict['conv_sents']
        # print('')

    # 手语翻译的评估
    # assert len(all_txt_outputs) == len(loader)

    # decode back to symbols
    decoded_txt = loader.dataset.arrays_to_sentences(arrays=all_txt_outputs)
    data_txt = loader.dataset.arrays_to_sentences(arrays=ref_text)
    # evaluate with metric on full dataset
    level = "word"
    join_char = " " if level in ["word", "bpe"] else ""
    # Construct text sequences for metrics
    txt_ref = [join_char.join(t) for t in data_txt]
    txt_hyp = [join_char.join(t) for t in decoded_txt]
    # post-process
    # if level == "bpe":
    #     txt_ref = [bpe_postprocess(v) for v in txt_ref]
    #     txt_hyp = [bpe_postprocess(v) for v in txt_hyp]
    assert len(txt_ref) == len(txt_hyp)

    # TXT Metrics
    txt_bleu = bleu(references=txt_ref, hypotheses=txt_hyp)
    txt_chrf = chrf(references=txt_ref, hypotheses=txt_hyp)
    txt_rouge = rouge(references=txt_ref, hypotheses=txt_hyp)

    recoder.print_log(f"Epoch {epoch}, {mode} bleu1 {txt_bleu['bleu1']: 2.2f} bleu2 {txt_bleu['bleu2']: 2.2f} bleu3 {txt_bleu['bleu3']: 2.2f} bleu4 {txt_bleu['bleu4']: 2.2f} chrf {txt_chrf: 2.2f} rouge {txt_rouge: 2.2f}", f"{work_dir}/{mode}.txt")

        
    # 手语识别的评估，使用sclite
    python_eval = True if evaluate_tool == "python" else False
    write2file(work_dir + "output-hypothesis-{}.ctm".format(mode), total_info, total_sent)
    write2file(work_dir + "output-hypothesis-{}-conv.ctm".format(mode), total_info,
               total_conv_sent)
    # conv_ret = evaluate(
    #     prefix=work_dir, mode=mode, output_file="output-hypothesis-{}-conv.ctm".format(mode),
    #     evaluate_dir=cfg.dataset_info['evaluation_dir'],
    #     evaluate_prefix=cfg.dataset_info['evaluation_prefix'],
    #     output_dir="epoch_{}_result/".format(epoch),
    #     python_evaluate=python_eval,
    # )
    lstm_ret = evaluate(
        prefix=work_dir, mode=mode, output_file="output-hypothesis-{}.ctm".format(mode),
        evaluate_dir=cfg.dataset_info['evaluation_dir'],
        evaluate_prefix=cfg.dataset_info['evaluation_prefix'],
        output_dir="epoch_{}_result/".format(epoch),
        python_evaluate=python_eval,
        triplet=True,
    )

    recoder.print_log(f"Epoch {epoch}, {mode} {lstm_ret: 2.2f}%", f"{work_dir}/{mode}.txt")

    return lstm_ret


def seq_feature_generation(loader, model, device, mode, work_dir, recoder):
    model.eval()

    src_path = os.path.abspath(f"{work_dir}{mode}")
    tgt_path = os.path.abspath(f"./features/{mode}")
    if not os.path.exists("./features/"):
        os.makedirs("./features/")

    if os.path.islink(tgt_path):
        curr_path = os.readlink(tgt_path)
        if work_dir[1:] in curr_path and os.path.isabs(curr_path):
            return
        else:
            os.unlink(tgt_path)
    else:
        if os.path.exists(src_path) and len(loader.dataset) == len(os.listdir(src_path)):
            os.symlink(src_path, tgt_path)
            return

    for batch_idx, data in tqdm(enumerate(loader)):
        recoder.record_timer("device")
        vid = device.data_to_device(data[0])
        vid_lgt = device.data_to_device(data[1])
        with torch.no_grad():
            ret_dict = model(vid, vid_lgt)
        if not os.path.exists(src_path):
            os.makedirs(src_path)
        start = 0
        for sample_idx in range(len(vid)):
            end = start + data[3][sample_idx]
            filename = f"{src_path}/{data[-1][sample_idx].split('|')[0]}_features.npy"
            save_file = {
                "label": data[2][start:end],
                "features": ret_dict['framewise_features'][sample_idx][:, :vid_lgt[sample_idx]].T.cpu().detach(),
            }
            np.save(filename, save_file)
            start = end
        assert end == len(data[2])
    os.symlink(src_path, tgt_path)


def write2file(path, info, output):
    filereader = open(path, "w")
    for sample_idx, sample in enumerate(output):
        for word_idx, word in enumerate(sample):
            filereader.writelines(
                "{} 1 {:.2f} {:.2f} {}\n".format(info[sample_idx],
                                                 word_idx * 1.0 / 100,
                                                 (word_idx + 1) * 1.0 / 100,
                                                 word[0]))

