# from turtle import distance
# import Levenshtein

import sys
sys.path.append('../..')
print(sys.path)

from signjoey.metrics import bleu, chrf, rouge, wer_list
import matplotlib.pyplot as plt
import numpy as np


train = open('train.txt','r')
test = open('test.txt','r')
dev = open('dev.txt','r')

train_question_list = []
test_question_list = []
dev_question_list = []
train_text_list = []
test_text_list = []
dev_text_list = []
train_question_word_list = []
test_question_word_list = []
dev_question_word_list = []

for line in train.readlines():
    line = line.replace('\n', '')
    folderAndTextLabel, text, label = line.split('%')
    textLabel= ' '.join(folderAndTextLabel.split(' '))
    train_text_list.append(textLabel)
    train_question_list.append(text)
    for i in text.split(' '):
        train_question_word_list.append(i)
    
for line in test.readlines():
    line = line.replace('\n', '')
    folderAndTextLabel, text, label = line.split('%')
    textLabel= ' '.join(folderAndTextLabel.split(' '))
    test_text_list.append(textLabel)
    test_question_list.append(text)
    for i in text.split(' '):
        test_question_word_list.append(i)

for line in dev.readlines():
    line = line.replace('\n', '')
    folderAndTextLabel, text, label = line.split('%')
    textLabel= ' '.join(folderAndTextLabel.split(' '))
    dev_text_list.append(textLabel)
    dev_question_list.append(text)
    for i in text.split(' '):
        dev_question_word_list.append(i)

# res_word_dev_list= [v for v in dev_question_word_list if v in train_question_word_list]

# res_word_test_list= [v for v in test_question_word_list if v in train_question_word_list]
# distance_dev = 0
# for question in dev_question_list:
#     min_distance = len(question)
#     for question_train in train_question_list:
#         min_distance = min(Levenshtein.distance(question_train,question),min_distance)
#     distance_dev = distance_dev + min_distance

# print(distance_dev / len(dev_question_list))
        

# distance_test = 0
# for question in test_question_list:
#     min_distance = len(question)
#     for question_train in train_question_list:
#         min_distance = min(Levenshtein.distance(question_train,question),min_distance)
#     distance_test = distance_test + min_distance

# print(distance_test/ len(test_question_list))


# distance_qa_train = 0
# len_train_sum = 0
# for i in range(0,len(train_text_list)):
#     distance_qa_train = distance_qa_train + Levenshtein.distance(train_text_list[i],train_question_list[i])

# distance_qa_dev = 0
# len_dev_sum = 0
# for i in range(0,len(dev_text_list)):
#     distance_qa_dev = distance_qa_dev + Levenshtein.distance(dev_text_list[i],dev_question_list[i])

# distance_qa_test = 0
# for i in range(0,len(test_text_list)):
#     distance_qa_test = distance_qa_test + Levenshtein.distance(test_text_list[i],test_question_list[i])

# distance_qa_train_mean = distance_qa_train/len(train_text_list)
# distance_qa_dev_mean = distance_qa_dev/len(dev_text_list)
# distance_qa_test_mean = distance_qa_test/len(test_text_list)

# txt_question_dev = []
# for i in dev_question_list:

# txt_answer_dev= []

txt_bleu_dev = bleu(references=dev_text_list, hypotheses=dev_question_list)
txt_bleu_test = bleu(references=test_text_list, hypotheses=test_question_list)
# txt_bleu_train = bleu(references=train_text_list, hypotheses=train_question_list)

bleu_dev_list = [txt_bleu_dev['bleu1'],txt_bleu_dev['bleu2'],txt_bleu_dev['bleu3'],txt_bleu_dev['bleu4']]
bleu_test_list = [txt_bleu_test['bleu1'],txt_bleu_test['bleu2'],txt_bleu_test['bleu3'],txt_bleu_test['bleu4']]

labels= ['BLEU-1','BLEU-2','BLEU-3','BLEU-4']

x = np.arange(len(labels))  # the label locations
width = 0.4  # the width of the bars
fig, ax = plt.subplots()
ax.bar(x - width/3 , bleu_dev_list, width/2, label = 'Dev')
ax.bar(x + width/3 , bleu_test_list, width/2, label = 'Test')
ax.set_ylabel('Scores')
# ax.set_title('Scores by group and gender')
ax.set_xticks(x)
ax.set_xticklabels(labels)

for i,bleu_n in enumerate(bleu_dev_list):
    ax.text(i - width/3, bleu_dev_list[i]+0.3,'{:.2f}'.format(bleu_dev_list[i]),ha='center')

for i,bleu_n in enumerate(bleu_test_list):
    ax.text(i + width/3 ,bleu_test_list[i]+0.3,'{:.2f}'.format(bleu_test_list[i]),ha='center')



ax.legend()

plt.show()
plt.savefig('dev_test_overfit.png')

train.close()
test.close()
dev.close()