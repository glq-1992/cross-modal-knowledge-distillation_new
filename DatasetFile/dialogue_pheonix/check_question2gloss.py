# 尝试把question转换为gloss（小写化）的序列
import sys # 导入上级目录
sys.path.append('../..')   
from pytorch_pretrained_bert.tokenization import BertTokenizer
tokenizer = BertTokenizer.from_pretrained(
                'bert-base-german-cased-v1',
                cache_dir='/disk1/shipeng/slrBertPheonix' + '/.pretrained_model_{}'.format(0))

train_file = open('train.txt','r')
gloss_list = []
question_list = []
question_gloss_list = []
gloss_dict = dict()
for line in train_file.readlines():
    folderAndTextLabel, question, gloss = line.strip().split('%')
    folder = folderAndTextLabel.split(' ')[0]
    TextLabel = ' '.join(folderAndTextLabel.split(' ')[1:])
    gloss_list.append(gloss)
    print(' '.join(tokenizer.tokenize(question)))
    for oneGloss in gloss.split(' '):
        gloss_dict[oneGloss.lower()] = oneGloss
    question_list.append(question)

for q in question_list:
    question_gloss = []
    q = q.lower()
    for word in q.split(' '):
        if word in gloss_dict.keys():
            question_gloss.append(gloss_dict[word])
    question_gloss_list.append(' '.join(question_gloss))



tokenizer.tokenize('')
print('----')