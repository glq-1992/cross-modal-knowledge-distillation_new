import re
import os
import cv2
import pdb
import glob
import pandas
import argparse
import numpy as np
from tqdm import tqdm
from functools import partial
from multiprocessing import Pool


def csv2dict(anno_path, dataset_type):
    inputs_list = pandas.read_csv(anno_path)
    if dataset_type == 'train':
        broken_data = [2390]
        inputs_list.drop(broken_data, inplace=True)
    inputs_list = (inputs_list.to_dict()['id|folder|signer|annotation'].values())
    info_dict = dict()
    info_dict['prefix'] = anno_path.rsplit("/", 3)[0] + "/features/fullFrame-210x260px"
    print(f"Generate information dict from {anno_path}")
    for file_idx, file_info in tqdm(enumerate(inputs_list), total=len(inputs_list)):
        fileid, folder, signer, label = file_info.split("|")
        num_frames = len(glob.glob(f"{info_dict['prefix']}/{dataset_type}/{folder}"))
        info_dict[file_idx] = {
            'fileid': fileid,
            'folder': f"{dataset_type}/{folder}",
            'signer': signer,
            'label': label,
            'num_frames': num_frames,
            'original_info': file_info,
        }
    return info_dict


def generate_gt_stm(md, info, save_path):
    if md == 'train':
        with open(save_path, "w") as f:
            for k, v in info.items():
                if not isinstance(k, int):
                    continue
                f.writelines(f"{v['folder']} 1 signer 0.0 1.79769e+308 {v['label']}\n")
    else:
        with open(save_path, "w") as f:
            for k, v in info.items():
                if not isinstance(k, int):
                    continue
                f.writelines(f"{v['folder']} 1 signer 0.0 1.79769e+308 {v['label']}\n")


def sign_dict_update(total_dict, info):
    for k, v in info.items():
        if not isinstance(k, int):
            continue
        split_label = v['label'].split()
        for gloss in split_label:
            if gloss not in total_dict.keys():
                total_dict[gloss] = 1
            else:
                total_dict[gloss] += 1
    return total_dict


def resize_img(img_path, dsize='210x260px'):
    dsize = tuple(int(res) for res in re.findall("\d+", dsize))
    img = cv2.imread(img_path)
    img = cv2.resize(img, dsize, interpolation=cv2.INTER_LANCZOS4)
    return img


def resize_dataset(video_idx, dsize, info_dict):
    info = info_dict[video_idx]
    img_list = glob.glob(f"{info_dict['prefix']}/{info['folder']}")
    for img_path in img_list:
        rs_img = resize_img(img_path, dsize=dsize)
        rs_img_path = img_path.replace("210x260px", dsize)
        rs_img_dir = os.path.dirname(rs_img_path)
        if not os.path.exists(rs_img_dir):
            os.makedirs(rs_img_dir)
            cv2.imwrite(rs_img_path, rs_img)
        else:
            cv2.imwrite(rs_img_path, rs_img)


def run_mp_cmd(processes, process_func, process_args):
    with Pool(processes) as p:
        outputs = list(tqdm(p.imap(process_func, process_args), total=len(process_args)))
    return outputs


def run_cmd(func, args):
    return func(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Data process for Visual Alignment Constraint for Continuous Sign Language Recognition.')
    parser.add_argument('--dataset', type=str, default='dialogue_pheonix',
                        help='save prefix')
    parser.add_argument('--dataset_root', type=str, default='/disk2/dataset_glq/CSL-TJU_image_112',
                        help='path to the dataset')
    parser.add_argument('--annotation_prefix', type=str, default='annotations/manual/{}.corpus.csv',
                        help='annotation prefix')
    parser.add_argument('--output_res', type=str, default='112x112px',
                        help='resize resolution for image sequence')
    parser.add_argument('--process_image', '-p', action='store_true',
                        help='resize image')
    parser.add_argument('--multiprocessing', '-m', action='store_true',
                        help='whether adopts multiprocessing to accelate the preprocess')

    args = parser.parse_args()
    mode = ["test", "train","dev"]
    sign_dict = dict()
    # aa=os.path.exists(f"./{args.dataset}")
    # aaa=f"./{args.dataset}"
    # print(aa)
    if not os.path.exists(f"./{args.dataset}"):
        os.makedirs(f"./{args.dataset}")
    for md in mode:
        # generate information dict
        data_txt = f"../DatasetFile/dialogue_pheonix/{md}.txt"
        information = {'prefix':args.dataset_root}
        with open(data_txt) as f:
            if md == 'train':
                for i, line in enumerate(f.readlines()):
                    line = line.replace('\n', '')
                    folderAndTextLabel, text, label = line.split('%')
                    folder = folderAndTextLabel.split(' ')[0]
                    information[i] = dict(folder=folder,label=label,original_info=line)
            else:
                for i, line in enumerate(f.readlines()):
                    line = line.replace('\n', '')
                    folderAndTextLabel, text, label = line.split('%')
                    folder = folderAndTextLabel.split(' ')[0]
                    # folderAndLabel, text = line.split('%')
                    # folder = folderAndLabel.split(' ')[0]
                    # label = ' '.join(folderAndLabel.split(' ')[1:])
                    information[i] = dict(folder=folder,label=label,original_info=line)

        np.save(f"./{args.dataset}/{md}_info.npy", information)
        # update the total gloss dict
        sign_dict_update(sign_dict, information)
        # generate groudtruth stm for evaluation
        generate_gt_stm(md, information, f"./{args.dataset}/{args.dataset}-groundtruth-{md}.stm")
        # resize images
        video_index = np.arange(len(information) - 1)
        # print(f"Resize image to {args.output_res}")
        # if args.process_image:
        #     if args.multiprocessing:
        #         run_mp_cmd(10, partial(resize_dataset, dsize=args.output_res, info_dict=information), video_index)
        #     else:
        #         for idx in tqdm(video_index):
        #             run_cmd(partial(resize_dataset, dsize=args.output_res, info_dict=information), idx)
    sign_dict = sorted(sign_dict.items(), key=lambda d: d[0])
    save_dict = {}
    for idx, (key, value) in enumerate(sign_dict):
        save_dict[key] = [idx + 1, value]
    np.save(f"./{args.dataset}/gloss_dict.npy", save_dict)
