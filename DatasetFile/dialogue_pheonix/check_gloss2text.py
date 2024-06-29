# 尝试把question转换为gloss（小写化）的序列
import sys # 导入上级目录
sys.path.append('../..')   
from pytorch_pretrained_bert.tokenization import BertTokenizer
tokenizer = BertTokenizer.from_pretrained(
                'bert-base-german-cased-v1',
                cache_dir='/disk1/shipeng/slrBertPheonix' + '/.pretrained_model_{}'.format(0))

train_file = open('train.txt','r')

for line in train_file.readlines():
    folderAndTextLabel, question, gloss = line.strip().split('%')
    folder = folderAndTextLabel.split(' ')[0]
    text = ' '.join(folderAndTextLabel.split(' ')[1:])
    gloss_lower = gloss.lower()
    tokenized_gloss = tokenizer.tokenize(gloss_lower)
    print('text: '+ text + ' | gloss: '+ gloss + ' | tokenized_gloss: ' + ' '.join(tokenized_gloss))

print('----')