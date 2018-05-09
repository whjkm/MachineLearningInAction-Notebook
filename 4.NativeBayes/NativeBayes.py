def loadDataSet():
    postingList = [['my','dog','has','flea','problem','help','please'],
                   ['maybe','not','take','him','to','dog','park','stupid'],
                   ['my','dalmation','is','so','cute','I','love','him'],
                   ['stop','posting','stupid','worthless','garbage'],
                   ['mr','licks','ate','my','steak','how','to','stop','him'],
                   ['quit','buying','worthless','dog','food','stupid']]
    classVec = [0, 1, 0, 1, 0, 1] #1代表侮辱性文字，0代表正常言论
    return postingList, classVec

def createVocabList(dataSet):
    vocabSet = set()    #创建一个空集
    for document in dataSet:
        vocabSet = vocabSet | set(document)  #创建两个集合的并集
    return  list(vocabSet)

#遍历查看该单词是否出现，出现该单词则将该单词置1
def setOfWordsVec(vocabList, inputSet):
    returnVec = [0] * len(vocabList)   #创建一个其中所含元素都为0的向量
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1    #索引位置
        else:
            print("the word: %s is not in my Vocabulary!" % word)
    return returnVec

#朴素贝叶斯分类器训练函数
from numpy import *
def trainNB0(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])       #计算每篇文档的词条数
    #侮辱性文件出现的概率，这个例子只有两个分类，非侮辱性概率 = 1- 侮辱性的概率
    #侮辱性文件的个数除以文件总数 = 侮辱性文件出现的概率
    pAbusive = sum(trainCategory) / float(numTrainDocs)
    #单词出现的次数
    p0Num = zeros(numWords)
    p1Num = zeros(numWords)
    #整个数据集中单词出现的次数
    p0Denom = 0.0
    p1Denom = 0.0

    #遍历所有的文件
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = p1Num/p1Denom
    p0Vect = p0Num/p0Denom
    return p0Vect, p1Vect, pAbusive

#优化版训练函数
def trainNB1(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory) / float(numTrainDocs)
    p0Num = ones(numWords)
    p1Num = ones(numWords)

    #整个数据集单词出现总数，2.0（主要是为了避免分母为0的情况）
    p0Denom = 2.0
    p1Denom = 2.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            #累加侮辱词的频次
            p1Num += trainMatrix[i]
            #对每篇文章的侮辱词的频次进行统计
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    #类别1，侮辱性文档的列表[log(P(F1|C1)...]
    p1Vect = log(p1Num / p1Denom)
    #类别0，正常文档的列表
    p0Vect = log(p0Num / p0Denom)
    return p0Vect, p1Vect, pAbusive

#朴素贝叶斯分类函数
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0

def testingNB():
    listOposts, listClasses = loadDataSet()
    myVocabList = createVocabList(listOposts)
    trainMat = []
    for postinDoc in listOposts:
        trainMat.append(setOfWordsVec(myVocabList, postinDoc))
    p0V, p1V, pAb = trainNB1(array(trainMat),array(listClasses))
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = array(setOfWordsVec(myVocabList, testEntry))
    print(testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb))
    testEntry = ['stupid', 'garbage']
    thisDoc = array(setOfWordsVec(myVocabList, testEntry))
    print(testEntry, 'classified as: ',classifyNB(thisDoc, p0V, p1V, pAb))


#朴素贝叶斯词袋模型
def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0] * len(vocabList)   #创建一个其中所含元素都为0的向量
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec








def test():
    listOPosts, listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setOfWordsVec(myVocabList,postinDoc))
    p0v,p1v,pab = trainNB0(trainMat, listClasses)
    print(p0v)
    print(p1v)
    print(pab)
    #print(setOfWordsVec(myVocabList, listOPosts[0]))



if __name__ == '__main__':
     #test()
     testingNB()







