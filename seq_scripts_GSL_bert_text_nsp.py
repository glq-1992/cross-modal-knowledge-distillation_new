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


def seq_train(loader, model, optimizer, device, epoch_idx, recoder):
    # model=device.model_to_device(model)
    model.train()
    loss_value = []
    clr = [group['lr'] for group in optimizer.optimizer.param_groups]
    for batch_idx, data in enumerate(loader):
        vid = device.data_to_device(data[0])
        vid_lgt = device.data_to_device(data[1])
        label = device.data_to_device(data[2])
        label_lgt = device.data_to_device(data[3])
        text = device.data_to_device(data[4])
        text_lgt = device.data_to_device(data[5])
        next_sentence_label = device.data_to_device(data[6])
        # info = data[6]
        # print(vid_lgt)
        ret_dict = model.forward_train(vid, vid_lgt, label, label_lgt, text, text_lgt,next_sentence_label)
        # print("1:{}".format(torch.cuda.memory_allocated(0)/1024/1024))

        loss = model.criterion_calculation(ret_dict, label, label_lgt,epoch_idx)
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
    model.eval()
    total_sent = []
    total_info = []
    total_conv_sent = []
    # stat = {i: [0, 0] for i in range(len(loader.dataset.dict))}
    sample_count = 0
    wer_sum = np.zeros([4])
    for batch_idx, data in enumerate(tqdm(loader)):
        recoder.record_timer("device")
        vid = device.data_to_device(data[0])
        vid_lgt = device.data_to_device(data[1])
        label = device.data_to_device(data[2])
        label_lgt = device.data_to_device(data[3])
        text = device.data_to_device(data[4])
        text_lgt = device.data_to_device(data[5])
        next_sentence_label = device.data_to_device(data[6])
        # info = data[6]
        
        with torch.no_grad():
            ret_dict = model.forward_test(vid, vid_lgt, label, label_lgt, text, text_lgt)
        
        total_info += [file_name for file_name in data[-1]]
        total_sent += ret_dict['recognized_sents']
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
