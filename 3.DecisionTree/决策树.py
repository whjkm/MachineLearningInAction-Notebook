#创建数据集
def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']  # 露出水面,脚蹼,表示数据集中特征的含义
    return dataSet, labels

#计算给定数据集的香农熵（经验熵）
from math import log
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)   #表示参与训练的数据量
    # 计算分类标签出现的次数
    labelCounts = {}
    for featVec in dataSet:
        # 存储当前的标签，每一行的最后一个数据表示的是标签
        currentLabel = featVec[-1]
        # 为所有的分类创建字典，记录当前类别出现的次数
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        # 使用所有类标签的发生频率计算类别出现的概率
        prob = float(labelCounts[key])/numEntries
        #计算香农熵，以2为底求对数，信息期望值
        shannonEnt -= prob * log(prob,2)
    return shannonEnt

from collections import Counter
def calcShannonEnt2(dataSet):
    # 统计标签出现的次数
    label_count = Counter(data[-1] for data in dataSet)
    # 计算概率
    probs = [p[1] / len(dataSet) for p in label_count.items()]
    # 计算香农熵
    shannonEnt = sum([-p * log(p,2) for p in probs])
    return shannonEnt

#遍历dataSet数据集，求出index对应的colnum列的值为value的行
#依据index列进行分类，如果index列的数据等于value的时候，就要将index划分到新创建的列表中
def splitDataSet(dataSet, index, value):
    #新建一个列表存储划分出来的数据
    retDataSet = []
    for featVec in dataSet:
        if featVec[index] == value:
            reducedFeatVec = featVec[:index]
            reducedFeatVec.extend(featVec[index+1:])
            #收集结果值index列为value的行,排除index列
            retDataSet.append(reducedFeatVec)
    #retDataSet = [data[:index] + data[index + 1:] for data in dataSet for i,v in enumerate(data) if i == index and v == value]
    return retDataSet

#选择最好的数据集划分方式
def chooseBestFeatureToSplit(dataSet):
    #求第一行有多少列的特征feature，label在最后一列
    numFeatures = len(dataSet[0]) - 1
    #计算整个数据集的原始香农熵
    baseEntropy = calcShannonEnt(dataSet)
    #最优的信息增益值，和最优的特征值的编号
    bestInfoGain = 0.0; bestFeature = -1
    #循环遍历数据集中的所有特征
    for i in range(numFeatures):
        #使用（List  Comprehension）来创建新的列表
        #获取数据集中所有的第i个特征值
        featList = [example[i] for example in dataSet]
        #获取去重之后的集合，使用set对list数据进行去重，set类型中每个值互不相同
        uniqueVals = set(featList)
        # 创建一个临时的信息熵
        newEntropy = 0.0
        #遍历某一列的value集合，计算该列的信息熵
        # 遍历当前特征中的所有唯一属性值，对每个唯一属性值划分一次数据集，计算数据集的新熵值，并对所有唯一特征值得到的熵求和
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i ,value)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)   #经验条件熵
        #信息增益：划分数据集前后的信息变化，获取信息熵最大的值
        #信息增益是熵的减少或者是数据无序度的减少
        infoGain = baseEntropy - newEntropy
        #比较所有特征中的信息增益，返回最好特征划分的索引值
        if(infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature

#选择出现次数最多的结果
import operator
def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    #排序得到出现次数最多的结果
    sortedClassCount = sorted(classCount.iteritems(), key = operator.itemgetter(1), reverse = True)
    return sortedClassCount[0][0]

#创建树
def createTree(dataSet, labels):
    #创建一个列表，包含所有的类标签（数据集的最后一列是标签）
    classList = [example[-1] for example in dataSet]
    #所有的类标签完全相同，则直接返回该类标签
    #列表中第一个值（标签）出现的次数==整个集合的数量，也就是说只有一个类别
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    #使用完了所有特征，仍然不能将数据集划分为仅包含唯一类别的分组。
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)  #挑选出现次数最多的类别作为返回值

    # 选择最优的列，得到最优列对应的label的含义
    bestFeat = chooseBestFeatureToSplit(dataSet)
    #获取label的名称
    bestFeatLabel = labels[bestFeat]
    #初始化myTree
    myTree = {bestFeatLabel:{}}
    #在标签列表中删除当前最优的标签
    del(labels[bestFeat])
    #得到最优特征包含的所有属性值
    featValues = [example[bestFeat] for example in dataSet]
    #去除重复的特征值
    uniqueVals = set(featValues)
    for value in uniqueVals:
        #求出剩余的标签label
        subLabels = labels[:]
        #遍历当前选择特征包含的所有属性值，在每个数据集划分上递归调用createTree()函数
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat,value), subLabels)
    return myTree

'''
args:
     inputTree: 已经训练好的决策树模型
     featLabels: Feature标签对应的名称，特征标签
     testVec: 测试输入的数据
'''
def classify(inputTree, featLabels, testVec):
    #获取tree的根节点对应的key值
    firstStr = list(inputTree.keys())[0]
    #通过key得到根节点对应的value
    secondDict = inputTree[firstStr]
    #获取根节点在label中的先后顺序
    featIndex = featLabels.index(firstStr)   #将标签字符串转换为索引位置
    for key in secondDict.keys():
        #如果到达叶子节点，则返回当前节点的分类标签
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                #判断节点，递归继续找
                classLabel = classify(secondDict[key], featLabels, testVec)
            else:
                classLabel = secondDict[key]
    return classLabel

    """
    key = testVec[featIndex]
    valueOfFeat = secondDict[key]
    if isinstance(valueOfFeat, dict):
        classLabel = classify(valueOfFeat, featLabels, testVec)
    else:
        classLabel = valueOfFeat
    """

#使用pickle模块存储决策树
def storeTree(inputTree, filename):
    import pickle
    fw = open(filename, 'w')
    pickle._dump(inputTree, fw)
    fw.close()

    '''
    with open(filename, 'wb') as fw:
        pickle.dump(inputTree, fw)
    '''

def grabTree(filename):
    import pickle
    fr = open(filename)
    return pickle.load(fr)

#对动物是否是鱼类分类的测试函数，并将结果使用matplotlib画出来
def fishTest():
    myDat, labels = createDataSet()
    import copy
    myTree = createTree(myDat, copy.deepcopy(labels))
    print(myTree)
    #[1,1]表示要取的分支上的节点位置，对应的结果值
    print(classify(myTree, labels, [1,1]))

    #画图可视化展现
    dtplot.createPlot(myTree)

def ContactLensesTest():
    #加载隐形眼镜相关的文本文件数据
    fr = open()
    #解析数据，获取features数据
    lenses = [inst.strip().split('\t') for inst in fr.readlines()]
    #得到数据相应的Labels
    lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
    #构造预测隐形眼镜的决策树
    lensesTree = createTree(lenses, lensesLabels)
    print(lensesTree)
    #画图可视化展现
    dtplot.createPlot(lensesTree)




def test():
    myDat,labels = createDataSet()
    print(myTree = createTree(myDat, labels))
    #print(calcShannonEnt(myDat))

if __name__ == '__main__':
     test()








