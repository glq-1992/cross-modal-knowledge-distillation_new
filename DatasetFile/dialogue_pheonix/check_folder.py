import glob
def getFrameNum(folder, root):
    img_folder =   + '/*.png'
    img_list_all = sorted(glob.glob(img_folder))