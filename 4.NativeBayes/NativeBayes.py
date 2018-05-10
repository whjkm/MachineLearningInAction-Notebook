import operator
import feedparser
import random


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
    p0Denom = 2.0  #拉普拉斯平滑
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
'''
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
'''

#文本解析，分词，解析为一共字符串列表
def textParse(bigString):
    import re
    #r表示raw String,自动将反斜杠转义
    listOfTokens = re.split(r'\W+', bigString)  #匹配除单词、数字外的任意字符串
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]

'''
#垃圾邮件测试函数
import random
def spamTest():
    docList =[]
    classList = []
    fullText = []
    for i in range(1,26):
        #切分解析文本数据
        #wordList = textParse(open('H:/python/email/spam/%d.txt' % i).read())
        #wordList = textParse(open('H:/python/email/spam/%d.txt' % i, 'r').read())
        try:
            wordList = textParse(open('H:/python/email/spam/{}.txt'.format(i)).read())
        except:
            wordList = textParse(open('H:/python/email/spam/{}.txt'.format(i),encoding = 'Windows 1252').read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        #wordList = textParse(open('H:/python/email/ham/%d.txt' % i).read())
        #wordList = textParse(open('H:/python/email/ham/%d.txt' % i, 'r').read())
        try:
            wordList = textParse(open('H:/python/email/ham/{}.txt'.format(i)).read())
        except:
            wordList = textParse(open('H:/python/email/ham/{}.txt'.format(i), encoding='Windows 1252').read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    # 创建词汇表
    vocabList = createVocabList(docList)
    # 训练数据集，总共50封邮件
    trainingSet = list(range(50))  #range返回的是range对象，不返回数组对象,不用list后面删除会出错
    testSet = []
    #随机取10封邮件用来进行测试
    for i in range(10):
        # random.uniform(x, y) 随机生成一个范围为 x - y 的实数
        randIndex = int(random.uniform(0,len(trainingSet)))
        #留存交叉验证
        testSet.append(trainingSet[randIndex])   #添加到测试集
        del(trainingSet[randIndex])    #在训练集删除添加到测试集中的数据
    trainMat = []
    trainClasses = []
    #用训练集训练
    for docIndex in trainingSet:
        trainMat.append(setOfWordsVec(vocabList,docList[docIndex]))  #构建词向量
        trainClasses.append(classList[docIndex])
    p0V, p1V, pSpam = trainNB1(array(trainMat), array(trainClasses))
    errorCount = 0    #错误个数
    #进行测试
    for docIndex in testSet:
        wordVector = setOfWordsVec(vocabList, docList[docIndex])
        if classifyNB(array(wordVector), p0V,p1V,pSpam) != classList[docIndex]:
            errorCount += 1  #分类错误数+1
    print('the errorCount is: ', errorCount)
    print('the testSet length is: ', len(testSet))
    print('the error rate is:', float(errorCount)/len(testSet))
'''

#朴素贝叶斯词袋模型
def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0] * len(vocabList)   #创建一个其中所含元素都为0的向量
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec


#RSS源分类器及高频词去除函数
def calcMostFreq(vocabList, fullText):
    #遍历词汇表中的每个词并统计它在文本中出现的次数
    freqDict = {}
    for token in vocabList:
        freqDict[token] = fullText.count(token) #统计每个词在文本中出现的次数
    #根据出现次数从高到低对词典进行排序，最后返回排序最高的30个单词
    sortedFreq = sorted(freqDict.items(), key = operator.itemgetter(1), reverse = True) #True表示降序排列
    return sortedFreq[0:30]



def localWords(feed1, feed0):
    docList = []
    classList = []
    fullText = []
    minLen = min(len(feed1['entries']), len(feed0['entries']))
    for i in range(minLen):
        wordList = textParse(feed1['entries'][i]['summary']) #每次访问一条RSS源
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(feed0['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)
    top30Words = calcMostFreq(vocabList, fullText)
    for pairW in top30Words:
        if pairW[0] in vocabList:
            vocabList.remove(pairW[0])   #去除出现次数最高的那些词
    trainingSet = list(range(2 * minLen))
    testSet = []
    import random
    for i in range(20):
        rangeIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[rangeIndex])
        del(trainingSet[rangeIndex])
    #testSet = [int(num) for num in random.sample(range(2 * minLen),20)]
    #trainingSet = list(set(range(2 * minLen)) - set(testSet))
    trainMat = []
    trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V, p1V, pSpam = trainNB1(array(trainMat), array(trainClasses))
    errorCount = 0
    for docIndex in testSet:
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
        if classifyNB(array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
    print('the error rate is: ', float(errorCount) / len(testSet))
    return vocabList, p0V, p1V



def getTopWords():
    ny = feedparser.parse('http://newyork.craigslist.org/stp/index.rss')
    sf = feedparser.parse('http://sfbay.craigslist.org/stp/index.rss')
    vocabList, p0V, p1V = localWords(ny,sf)
    topNY = []
    topSF = []
    for i in range(len(p0V)):
        if p0V[i] > -6.0:
            topSF.append((vocabList[i],p0V[i]))
        if p1V[i] > -6.0:
            topNY.append((vocabList[i],p1V[i]))
    sortedSF = sorted(topSF, key=lambda pair: pair[1], reverse=True)
    print("SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**")
    for item in sortedSF:
        print(item[0])
    sortedNY = sorted(topNY, key=lambda pair: pair[1], reverse=True)
    print("NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**")
    for item in sortedNY:
        print(item[0])


'''
def testRss():
    ny = feedparser.parse('http://newyork.craigslist.org/stp/index.rss')
    sf = feedparser.parse('http://sfbay.craigslist.org/stp/index.rss')
    #print(ny)
    #print(sf)
    vocabList, p0V, p1V = localWords(ny, sf)
    print(vocabList)
    print(p0V)
    print(p1V)
'''


'''
#字符串切分
def test1():
    mySent = 'This book is the best book on Python or M.L. I have ever laid eyes upon.'
    print(mySent.split())
    #使用正则表达式切分
    import re
    regEx = re.compile('\\W*')
    listOfTokens = regEx.split(mySent)
    print(listOfTokens)


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
'''


if __name__ == '__main__':
     #test()
     #testingNB()
     #test1()
     #spamTest()
     #testRss()
     getTopWords()






