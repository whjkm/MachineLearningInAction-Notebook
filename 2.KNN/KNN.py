#导入科学计算包
from numpy import *
#导入运算符模块
import operator
import os

#创建数据集和标签
def createDataSet():
    group = array([[1.0,1.1],
                  [1.0,1.0],
                  [0,0],
                  [0,0.1]])
    labels = ['A','A','B','B']
    return group, labels

'''
Parameters:
    inX - 用于分类的数据(测试集)
    dataSet - 用于训练的数据(训练集)
    labes - 训练数据集的label
    k - 选择距离最小的k个点
return：
    sortedClassCount[0][0] - 输入数据的预测分类
'''
# k-近邻算法
def classify0(inX, dataSet, labels, k):
    # 计算距离
    dataSetSize = dataSet.shape[0]
    # 用tile将输入向量复制成和数据集一样大的矩阵
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    # 将矩阵的每一行相加,axis = 1表示行相加
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    # 按距离从小到大排序，并返回对应的索引位置
    sortedDistIndicies = distances.argsort()

    # 创建一个字典,存储标签和出现次数
    classCount = {}
    # 选择距离最小的k个点
    for i in range(k):
        # 查找样本的标签类型
        voteIlabel = labels[sortedDistIndicies[i]]
        # 在字典中给找到的样本标签类型+1
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
        # 排序并返回出现次数最多的标签类型
        sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1), reverse=True)
        return sortedClassCount[0][0]

#实现 classify0()的第二种方式
def classify1(inX, dataSet, Labels, k):
    #计算距离
    #import numpy as np\
    import collections
    dist = np.sum((inX - dataSet)**2, axis=1) ** 0.5  #利用numpy中的broadcasting
    #k个最近的标签
    k_labels = [Labels[index] for index in dist.argsort()[0 : k]]
    #出现次数最多的标签即为最终类别
    label = collections.Counter(k_labels).most_common(1)[0][0]
    return label

# 测试样例
'''
def test1():
    group, labels = createDataSet()
    print(str(group))
    print(str(labels))
    print(classify0([0.1, 0.1], group, labels, 3))
'''

#将文本记录转化为NumPy矩阵
def file2matrix(filename):
    fr = open(filename,'r')
    #获取文件数据行的行数
    #arrayOLines = fr.readlines()
    #numberOfLines = len(arrayOLines)
    numberOfLines = len(fr.readlines())
    #生成一个0矩阵
    returnMat = zeros((numberOfLines,3))
    #要返回的标签
    classLabelVector = []
    fr = open(filename, 'r')
    index = 0
    #解析文件数据到列表
    for line in fr.readlines():
        # 去除字符串首尾的空格
        line = line.strip()
        #用制表符\t分割字符串
        listFormLine = line.split('\t')
        #每列的属性数据
        returnMat[index] = listFormLine[0:3]
        #每列的label标签数据,-1最后一列
        classLabelVector.append(int(listFormLine[-1]))
        index += 1
    return  returnMat,classLabelVector

#归一化特征值
def autoNorm(dataSet):
    #每列的最小值
    minVals = dataSet.min(0)
    #每列的最大值
    maxVals = dataSet.max(0)
    #归一化处理的范围
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    #生成与最小值之差组成的矩阵
    normDataSet = dataSet - tile(minVals,(m,1))
    #最小值之差除以最大值和最小值的差值
    normDataSet = normDataSet / tile(ranges,(m,1))
    # norm_dataset = (dataset - minvalue) / ranges
    return  normDataSet, ranges, minVals

#测试算法
def datingClassTest():
    #测试范围，一部分测试一部分作为样本
    hoRatio = 0.1
    #加载数据
    datingDataMat, datingLabels = file2matrix("datingTestSet2.txt")
    #归一化数据
    normMat, ranges, minVals = autoNorm(datingDataMat)
    #数据的行数
    m = normMat.shape[0]
    #设置样本的测试数据
    numTestVecs = int(m * hoRatio)
    print('numTestVecs', numTestVecs)
    #分类错误数
    errorCount = 0
    #numTestVecs: m表示训练样本的数量
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i], normMat[numTestVecs : m], datingLabels[numTestVecs : m], 3)
        print("the classifier came back with: %d, the real answer is : %d" %(classifierResult, datingLabels[i]))
        errorCount += classifierResult != datingLabels[i]
    print("the total error rate is :%f" %(errorCount / numTestVecs))
    print(errorCount)

def classifyPerson():
    resultList = ['not at all', 'in small doses', 'in large doses']
    percentTats = float(input("percentage of time spent playing video games?"))
    ffMiles = float(input("frequent filer miles earned per year?"))
    iceCream = float(input("liters of ice cream consumed per year?"))
    datingDataMat, datingLabel = file2matrix("datingTestSet2.txt")
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = array([ffMiles, percentTats, iceCream])
    classifierResult = classify0((inArr - minVals)/ranges, normMat, datingLabel, 3)
    print("You will probably like this person: ", resultList[classifierResult - 1])

def draw():
# 使用Matplotlib画二维散点图
    import matplotlib
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    datingDataMat, datingLabels = file2matrix("datingTestSet2.txt")
    ax.scatter(datingDataMat[:,1], datingDataMat[:,2])
    #ax.scatter(datingDataMat[:,1], datingDataMat[:,2], 15.0*array(datingLabels), 15,0*array(datingLabels))
    plt.show()


#将图像数据转换为向量
def img2vector(filename):
    returnVect = zeros((1, 1024))
    fr = open(filename, 'r')
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32 * i + j] = int(lineStr[j])
    return returnVect

def handwritingClassTest():
    #导入数据
    hwLabels = []
    trainingFileList = os.listdir('H:/机器学习/MachineLearning-master/MachineLearning-master/input/2.KNN/trainingDigits')
    m = len(trainingFileList)
    trainingMat = zeros((m, 1024))
    for i in range(m):
        #从文件名解析分类数字
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i] = img2vector('H:/机器学习/MachineLearning-master/MachineLearning-master/input/2.KNN/trainingDigits/%s' % fileNameStr)

    #导入测试数据
    testFileList = os.listdir('H:/机器学习/MachineLearning-master/MachineLearning-master/input/2.KNN/testDigits')
    errorCount = 0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('H:/机器学习/MachineLearning-master/MachineLearning-master/input/2.KNN/testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print("the classifier came back with: %d, the real answer is: %d" %(classifierResult, classNumStr))
        errorCount += classifierResult != classNumStr
    print("\nthe total number of errors is: %d" % errorCount)
    print("\nthe total error rate is: %f" %(errorCount / mTest))

if __name__ == '__main__':
    test1()
    #datingClassTest()
    #draw()
    #classifyPerson()
    #handwritingClassTest()
