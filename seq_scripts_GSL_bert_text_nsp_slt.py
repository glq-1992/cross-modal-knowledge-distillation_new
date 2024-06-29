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

def seq_train(loader, model, optimizer, device, epoch_idx, recoder):
    # model=device.model_to_device(model)
    model.train()
    # for name,params in model.named_parameters():
    #     if 'sign_encoder' in name or 'bert_text' in name or 'multimodal_fusion' in name:
    #         print(name)
    #         params.requires_grad = False
    loss_value = []
    clr = [group['lr'] for group in optimizer.optimizer.param_groups]
    for batch_idx, data in enumerate(loader):
        vid = device.data_to_device(data[0])
        vid_lgt = device.data_to_device(data[1])
        label = device.data_to_device(data[2])
        label_lgt = device.data_to_device(data[3])
        label_text = device.data_to_device(data[4])
        label_text_lgt = device.data_to_device(data[5])
        label_text_target = device.data_to_device(data[6])
        text = device.data_to_device(data[7])
        text_lgt = device.data_to_device(data[8])
        next_sentence_label = device.data_to_device(data[9])
        # info = data[6]
        # print(vid_lgt)
        ret_dict = model.forward_train(vid, vid_lgt, label, label_lgt,label_text,label_text_lgt, text, text_lgt,next_sentence_label)
        # print("1:{}".format(torch.cuda.memory_allocated(0)/1024/1024))

        loss = model.criterion_calculation(ret_dict, label, label_lgt, label_text_target, label_text_lgt, epoch_idx)
        # indexOfChongfu = 118
        # if indexOfChongfu in label:
        #     print(data[-1])
        if np.isinf(loss.item()) or np.isnan(loss.item()):
            # print(loss.item())
            print(data[-1])
            print('loss is nan or inf!')
            # if epoch_idx == 0:
            #     f = open("pheonixT_ctc_nan.txt", "a")
            #     f.write(str(data[-1]) + '\n')
            #     f.close()
            continue
        
        optimizer.zero_grad()
        loss.backward()
        # nn.utils.clip_grad_norm_(model.rnn.parameters(), 5)
        optimizer.step()
        loss_value.append(loss.item())
        if batch_idx % recoder.log_interval == 0:
            recoder.print_log(
                '\tEpoch: {}, Batch({}/{}) done. Loss: {:.8f}  lr:{:.8f}'
                    .format(epoch_idx, batch_idx, len(loader), loss.item(), clr[0]))
    optimizer.scheduler.step()
    recoder.print_log('\tMean training loss: {:.10f}.'.format(np.mean(loss_value)))
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
        label_text = device.data_to_device(data[4])
        label_text_lgt = device.data_to_device(data[5])
        label_text_target = device.data_to_device(data[6])
        text = device.data_to_device(data[7])
        text_lgt = device.data_to_device(data[8])
        next_sentence_label = device.data_to_device(data[9])
        # info = data[6]
        
        with torch.no_grad():
            ret_dict = model.forward_test(vid, vid_lgt, label, label_lgt,label_text,label_text_lgt, text, text_lgt,translation_beam_size,translation_beam_alpha)
        
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
    # filename = 'result_translation_baseline_dev.txt'
    # with open (filename, 'w') as file_object:
    #     for hyp_one in txt_hyp:
    #         file_object.write(hyp_one + "\n") 

    # filename_ref = 'result_translation_ref.txt'
    # with open (filename_ref, 'w') as file_object:
    #     for ref_one in txt_ref:
    #         file_object.write(ref_one + "\n") 

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

