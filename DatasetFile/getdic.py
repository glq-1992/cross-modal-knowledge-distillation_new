sourceFileTrain = open('/home/gaoliqing/shipeng/code/slrBert/DatasetFile/dialogue_pheonix/train.txt','r+')
sourceFileTest = open('/home/gaoliqing/shipeng/code/slrBert/DatasetFile/dialogue_pheonix/test.txt','r+')

sourceFileDev = open('/home/gaoliqing/shipeng/code/slrBert/DatasetFile/dialogue_pheonix/dev.txt','r+')

# dicFile = open('/home/gaoliqing/shipeng/code/slrBert/DatasetFile/dialogue_pheonix/vocab.txt',"w+")

dictSet = set()
for i in sourceFileTrain.readlines():
    wordList = i.strip().split('%')[-1].split(' ')
    for word in wordList:
        dictSet.add(word)
letterSet = set()
for i in dictSet:
    if '+' in i:
        print(i)
        for j in i.split('+'):
            letterSet.add(j)
for i in sourceFileTest.readlines():
    wordList = i.strip().split('%')[-1].split(' ')
    for word in wordList:
        dictSet.add(word)
for i in sourceFileDev.readlines():
    wordList = i.strip().split('%')[-1].split(' ')
    for word in wordList:
        dictSet.add(word)

# for i in dictSet:
#     dicFile.write(i+'\n')
sourceFileTrain.close()
sourceFileTest.close()
sourceFileDev.close()
# dicFile.close()