train = open('train.txt','r')
test = open('test.txt','r')
dev = open('dev.txt','r')

label_text_voc = set()
question_voc = set()
for line in train.readlines():
    line = line.replace('\n', '')
    folderAndTextLabel, text, label = line.split('%')
    folder = folderAndTextLabel.split(' ')[0]
    testLabel = ''.join(folderAndTextLabel.split(' ')[1:])
    for j in folderAndTextLabel.split(' ')[1:]:
        label_text_voc.add(j)
    for k in text.split(' '):
        question_voc.add(k)

for line in test.readlines():
    line = line.replace('\n', '')
    folderAndTextLabel, text, label = line.split('%')
    folder = folderAndTextLabel.split(' ')[0]
    testLabel = ''.join(folderAndTextLabel.split(' ')[1:])
    for j in folderAndTextLabel.split(' ')[1:]:
        label_text_voc.add(j)
    for k in text.split(' '):
        question_voc.add(k)

for line in dev.readlines():
    line = line.replace('\n', '')
    folderAndTextLabel, text, label = line.split('%')
    folder = folderAndTextLabel.split(' ')[0]
    testLabel = ''.join(folderAndTextLabel.split(' ')[1:])
    for j in folderAndTextLabel.split(' ')[1:]:
        label_text_voc.add(j)
    for k in text.split(' '):
        question_voc.add(k)

train.close()
test.close()
dev.close()