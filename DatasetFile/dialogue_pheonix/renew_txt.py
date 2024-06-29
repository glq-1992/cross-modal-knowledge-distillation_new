import pandas as pd

old_txt = open('/home/gaoliqing/shipeng/code/slrBert/DatasetFile/dialogue_pheonix/test_old.txt','r')
new_txt = open('/home/gaoliqing/shipeng/code/slrBert/DatasetFile/dialogue_pheonix/test.txt','w')

# official = open('/disk1/dataset/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/annotations/manual/PHOENIX-2014-T.dev.corpus.txt','r')

data = pd.read_csv('/disk1/dataset/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/annotations/manual/PHOENIX-2014-T.test.corpus.csv',sep='|')

folder_gloss = dict()
# for i in official.readlines():
#     folder = i.strip().split('|')[0]
#     gloss = i.strip().split('|')[-2]
#     folder_gloss[folder] = gloss
for i in range(0,data.shape[0]):
    folder_gloss[data['name'][i]] = data['orth'][i]

for i in old_txt.readlines():
    folder = i.strip().split(' ')[0].split('/')[-1]
    gloss = folder_gloss[folder]
    new_txt.write(i.strip()+'%'+gloss+'\n')

old_txt.close()
new_txt.close()
# official.close()