import numpy as np
import matplotlib.pyplot as plt
import random
from matplotlib.font_manager import FontProperties

# 读取数据
def loadDataSet():
    dataMat = []
    labelMat = []
    fr = open('H:/python/ml/testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()     #去除回车
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])]) #为了计算方便，在每一行添加一个X0 = 1.0
        labelMat.append(int(lineArr[2]))     #添加标签
    fr.close()
    return dataMat, labelMat

#logistic函数，输入可以是向量
def sigmoid(inX):
    return 1.0/(1+ np.exp(-inX))

#把数据集通过散点图绘制出来
def plotDataSet():
    dataMat, labelMat = loadDataSet()      #加载数据集
    dataArr = np.array(dataMat)            #转换成numpy的array数组
    n = np.shape(dataMat)[0]               #数据个数
    xcord1 =[]; ycord1 =[]                 #正样本
    xcord2 =[]; ycord2 =[]                 #负样本
    for i in range(n):                     #根据数据集标签进行分类
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i,1]); ycord1.append(dataArr[i,2])        # 1为正样本
        else:
            xcord2.append(dataArr[i,1]); ycord2.append(dataArr[i,2])        # 0为负样本
    fig = plt.figure()
    ax = fig.add_subplot(111)                                               # 添加subplot
    # s=20,默认为20，表示点的大小，marker = 's'，表示为正方形 ，alpha:是透明程度
    ax.scatter(xcord1, ycord1, s = 20, c = 'red', marker = 's', alpha = .5)   # 散点图，绘制正样本
    ax.scatter(xcord2, ycord2, s = 20, c = 'green', alpha = .5)
    plt.title('DataSet')
    plt.xlabel('x'); plt.ylabel('y')
    plt.show()

# 梯度上升优化算法
def gradAscent(dataMatIn, classLabels):
    dataMatrix = np.mat(dataMatIn)
    labelMat = np.mat(classLabels).transpose()
    m, n = np.shape(dataMatrix)
    alpha = 0.001           # 移动步长
    maxCycles = 500         # 最大迭代次数
    weights = np.ones((n, 1))          # n行1列，将每个回归系数初始化为1
    # weights_array = np.array([])

    # 对公式进行向量化  θ:=θ-α.x'.E
    for k in range(maxCycles):
        h = sigmoid(dataMatrix * weights)    # A=x.θ和g(A)
        error = (labelMat - h)      # E=g(A)-y  训练数据的损失
        weights = weights + alpha * dataMatrix.transpose() * error   # 梯度上升向量化公式
        # weights_array = np.append(weights_array, weights)
    # weights_array = weights_array.reshape(maxCycles, n)
    # return weights.getA(), weights_array        # 将numpy矩阵转换为数组
    return weights.getA()

#  绘制最佳拟合直线
def plotBestFit(weights):
    dataMat, labelMat = loadDataSet()
    dataArr = np.array(dataMat)             #转换为numpy的array数组
    n = np.shape(dataMat)[0]
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i,1])
            ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1])
            ycord2.append(dataArr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s = 20, c = 'red', marker = 's', alpha = .5)
    ax.scatter(xcord2, ycord2, s = 20, c = 'green', alpha = .5)
    x = np.arange(-3.0, 3.0, 0.1)    #以0.1为步长构造一个-3到3的array
    # w0+w1*x+w2*y=0 => y = (-w0-w1*x)/w2
    y = (-weights[0] - weights[1] * x) / weights[2]
    ax.plot(x, y)   #绘制直线
    plt.title('BestFit')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()


# 随机梯度上升算法
def stocGradAscent0(dataMatrix, classLabels):
    m, n = np.shape(dataMatrix)
    alpha = 0.01
    weights = np.ones(n)
    for i in range(m):
        # sum(dataMatrix[i]*weights)为了求 f(x)的值， f(x)=a1*x1+b2*x2+..+nn*xn
        h = sigmoid(sum(dataMatrix[i] * weights))   #h为一个具体的数值
        error = classLabels[i] - h
        weights = weights + alpha * error * dataMatrix[i]
    return  weights

# 改进的随机梯度上升算法
def stocGradAscent1(dataMatrix, classLabels, numIter = 150):
    m,n = np.shape(dataMatrix)
    weights = np.ones(n)
    # weights_array = np.array([])        # 存储每次更新的回归系数
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4 / (1.0 + j + i) + 0.01      # alpha每次迭代时需要调整, 降低alpha的大小，每次减少1/j+i
            randIndex = int(random.uniform(0, len(dataIndex)))   # 随机选择样本
            h = sigmoid(sum(dataMatrix[randIndex] * weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            # weights_array = np.append(weights_array, weights, axis=0)        # 添加回归系数到数组
            del(dataIndex[randIndex])       # 删除已经使用的样本
    # weights_array = weights_array.reshape(numIter*m, n)                      # 改变维度
    # return weights, weights_array
    return weights

# 绘制回归系数与迭代次数的关系
def plotWeights(weights_array1, weights_array2):
    # 设置汉字格式
    font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)
    # 将fig画布分隔成3行2列，不共享x轴和y轴， fig画布的大小为（20,10）
    fig, axs = plt.subplots(nrows=3, ncols=2, sharex=False, sharey=False, figsize=(20, 10))

    x1 = np.arange(0, len(weights_array1), 1)
    # 绘制w0与迭代次数的关系
    axs[0][0].plot(x1, weights_array1[:, 0])
    axs0_title_text = axs[0][0].set_title(u'梯度上升算法：回归系数与迭代次数关系', FontProperties=font)
    axs0_ylabel_text = axs[0][0].set_ylabel(u'W0', FontProperties=font)
    plt.setp(axs0_title_text, size=20, weight='bold', color='black')
    plt.setp(axs0_ylabel_text, size=20, weight='bold', color='black')
    # 绘制w1与迭代次数的关系
    axs[1][0].plot(x1, weights_array1[:, 1])
    axs1_ylabel_text = axs[1][0].set_ylabel(u'W1', FontProperties=font)
    plt.setp(axs1_ylabel_text, size=20, weight='bold', color='black')
    # 绘制w2与迭代次数的关系
    axs[2][0].plot(x1, weights_array1[:, 2])
    axs2_xlabel_text = axs[2][0].set_xlabel(u'迭代次数',FontProperties = font)
    axs2_ylabel_text = axs[2][0].set_ylabel(u'W1', FontProperties=font)
    plt.setp(axs2_xlabel_text, size=20, weight='bold', color='black')
    plt.setp(axs2_ylabel_text, size=20, weight='bold', color='black')

    x2 = np.arange(0, len(weights_array2), 1)
    # 绘制w0与迭代次数的关系
    axs[0][1].plot(x2, weights_array2[:, 0])
    axs0_title_text = axs[0][1].set_title(u'改进的随机梯度上升算法：回归系数与迭代次数关系', FontProperties=font)
    axs0_ylabel_text = axs[0][1].set_ylabel(u'W0', FontProperties=font)
    plt.setp(axs0_title_text, size=20, weight='bold', color='black')
    plt.setp(axs0_ylabel_text, size=20, weight='bold', color='black')
    # 绘制w1与迭代次数的关系
    axs[1][1].plot(x2, weights_array2[:, 1])
    axs1_ylabel_text = axs[1][1].set_ylabel(u'W1', FontProperties=font)
    plt.setp(axs1_ylabel_text, size=20, weight='bold', color='black')
    # 绘制w2与迭代次数的关系
    axs[2][1].plot(x2, weights_array2[:, 2])
    axs2_xlabel_text = axs[2][1].set_xlabel(u'迭代次数', FontProperties=font)
    axs2_ylabel_text = axs[2][1].set_ylabel(u'W1', FontProperties=font)
    plt.setp(axs2_xlabel_text, size=20, weight='bold', color='black')
    plt.setp(axs2_ylabel_text, size=20, weight='bold', color='black')

    plt.show()


# logistic回归分类函数
def classifyVector(inX, weights):
    prob = sigmoid(sum(inX*weights))
    if prob > 0.5:
        return 1.0
    else:
        return 0.0

def colicTest():
    frTrain = open('H:/python/ml/horseColicTraining.txt')
    frTest = open('H:/python/ml/horseColicTest.txt')
    trainingSet = []
    trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
    trainWeights = stocGradAscent1(np.array(trainingSet), trainingLabels, 500)
    errorCount = 0
    numTestVec = 0
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int (classifyVector(np.array(lineArr), trainWeights)) != int(currLine[21]):
            errorCount += 1
    errorRate = (float(errorCount)/numTestVec)
    print("the error rate of this test is : %f" % errorRate)
    return  errorRate

def multiTest():
    numTests = 10
    errorSum = 0.0
    for k in range(numTests):
        errorSum += colicTest()
    print("after %d iteration the average error rate is: %f" % (numTests, errorSum/float(numTests)))








if __name__ == '__main__':
    #Gradient_Ascent_test()
    #plotDataSet()
    dataMat, labelMat = loadDataSet()
    # weights = gradAscent(dataMat, labelMat)
    # weights = stocGradAscent0(np.array(dataMat), labelMat)
    # weights = stocGradAscent1(np.array(dataMat), labelMat)
    # weights1, weights_array1 = stocGradAscent1(np.array(dataMat), labelMat)
    # weights2, weights_array2 = gradAscent(dataMat, labelMat)
    # plotWeights(weights_array1, weights_array2)
    # plotBestFit(weights)




def Gradient_Ascent_test():
    def f_prime(x_old):    #f(x)的导数   f'(x) = -2x + 4
        return -2 * x_old +4
    x_old = -1     #初始值，给一个小于x_new的值
    x_new = 0      #初始值
    alpha = 0.01   #步长，学习速率
    presision = 0.00000001     #精度
    while abs(x_new - x_old) > presision:
        x_old = x_new;
        x_new = x_old + alpha * f_prime(x_old)   #迭代
    print(x_new)



