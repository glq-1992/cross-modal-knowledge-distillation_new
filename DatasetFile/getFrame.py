import glob
import os

from tqdm import tqdm

video_type_list=['Entity','Template','Response']
person_list=['P001','P002','P003','P004','P005','P006','P007','P008','P009','P010',] 

for video_name in video_type_list:
    for person_name in person_list:
        root_path='/disk2/dataset_glq/CSL-TJU_third_person_image_160/'+video_name+'/'+person_name
        frame_num=0
        video_list = os.listdir(root_path)
        for i in range(0,len(video_list)):
            video_path = video_list[i]
            video_path = os.path.join(root_path,video_path)
            frame_list = os.listdir(video_path)
            # print(frame_list.len())
            frame_num=frame_num+len(frame_list)

        print('VideoType: %s , Person: %s , num of frames : %d'%(video_name,person_name,frame_num))