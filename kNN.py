import numpy as np
import operator
from os import listdir
import matplotlib.pyplot as plt

def createDataSet():
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

# inX是要分类的输入向量，k选择最近邻居的数目
def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]

    # # 距离计算（欧氏距离公式）
    # np.tile(),inX在列方向上重复1次，行上重复dataSetSize次
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat**2  # 平方
    sqDistances = sqDiffMat.sum(axis=1)  # 求和
    distances = sqDistances**0.5  # 开方
    # argsort()从小到大排序 提取对应的索引
    sorteDistIndices = distances.argsort()
    # 选择距离最小的k个点
    classCount = {}
    for i in range(k):
        votelabel = labels[sorteDistIndices[i]] # 对应的标签
        # 计算该类别的票数
        # Python 字典(Dictionary) get() 函数返回指定键的值，
        # 如果值不在字典中返回默认值。
        classCount[votelabel] = classCount.get(votelabel, 0) + 1
    # 按字典第二个元素从大到小排序
    sortedClassCount = sorted(classCount.items(),  # 返回的是一个列表
                              key=operator.itemgetter(1),  # 根据数据的第二个成员排序
                              reverse=True)  # 降序排列
    return sortedClassCount[0][0]

# group, labels = createDataSet()
# print(classify0([0, 0], group, labels, 3))
###############################################################################

################################准备数据#######################################
# 将文本记录转换为NumPy的解析程序
def file2matrix(filename):
    fr = open(filename)
    # 逐行读取，存放到list中
    arrayOLines = fr.readlines()
    # 得到文件行数
    numberOfLines = len(arrayOLines)
    # 创建返回的NumPy矩阵，初始化为0
    returnMat = np.zeros((numberOfLines, 3))
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        line = line.strip()  # 截取掉所有的回车字符
        listFromLine = line.split('\t')  # 用'\t'分割成列表数据
        returnMat[index, :] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat, classLabelVector

datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
# print(datingDataMat)
# print(datingLabels[0:20])
########################################################################

#####################################分析数据###########################
# fig = plt.figure()
# ax = fig.add_subplot(111)
# # 使用datingDataMat矩阵的第二列、第三列数据
# ax.scatter(datingDataMat[:, 1], datingDataMat[:, 2],
#            15.0*np.array(datingLabels), 15.0*np.array(datingLabels))
# ax.axis([-2, 25, -0.2, 2.0]) # x轴范围，y轴范围
# ax.set_xlabel('Percentage of Time Spent Playing Video Games')
# ax.set_ylabel('Liters of Ice Cream Consumed Per Week')
# plt.show()
#########################################################################

###################################归一化数值############################
def autoNorm(dataSet):
    minVals = dataSet.min(0)  # 参数0表示取每一列的最小值
    maxVals = dataSet.max(0)  # 1*3
    ranges = maxVals - minVals  # 1*3
    normDataSet = np.zeros(dataSet.shape)  # 1000*3
    m = dataSet.shape[0]
    normDataSet = dataSet - np.tile(minVals, (m, 1))
    normDataSet = normDataSet/np.tile(ranges, (m, 1))
    return normDataSet, ranges, minVals

# normMat, ranges, minVals = autoNorm(datingDataMat)
# print(normMat[:20])
# print(ranges)
# print(minVals)
###################################################################

############################测试算法###############################
def datingClassTest():
    hoRatio = 0.10
    # 原始数据
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    # 归一化数据
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    # 测试数据的数量
    numTestVec = int(hoRatio*m)
    errorCount = 0.0
    for i in range(numTestVec):
        classfierResult = classify0(normMat[i], normMat[numTestVec:m, :],
                                    datingLabels[numTestVec:m], 3)
        print('the classfier came back with: %d, the real answer is: %d'
              % (classfierResult, datingLabels[i]))
        if (classfierResult != datingLabels[i]):
            errorCount += 1.0
    print('the total error rate is: %f' % (errorCount/float(numTestVec)))

# datingClassTest()
##########################################################################

###################################使用算法###############################
def classifyPerson():
    resultList = ['not at all', 'in small doses', 'in large doses']
    ffMiles = float(input('frequent flier miles earned per year?'))
    percentTats = float(input('percentage of time spent playing video games?'))
    iceCream = float(input('liters of ice cream consumed per year?'))
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, mixVals = autoNorm(datingDataMat)
    inArr = np.array([ffMiles, percentTats, iceCream])
    classifierReasult = classify0((inArr - mixVals)/ranges, normMat, datingLabels, 3)
    print('You will probably like this person: ', resultList[classifierReasult - 1])

# classifyPerson()
########################################################################################

#####################################将图像转换为测试向量###############################
# 32*32的二进制图像矩阵转换为1*1024的向量
def img2vector(filename):
    returnVect = np.zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32*i+j] = int(lineStr[j])
    return returnVect

# testVector = img2vector('digits/testDigits/0_13.txt')
# print(testVector[0, 0:32])
#######################################################################################

###################################手写数字测试算法####################################
def handwritingClassTest():
    hwLabels = []
    # listdir获取文件的目录内容
    trainingFileList = listdir('digits/trainingDigits')
    m = len(trainingFileList)
    trainingMat = np.zeros((m, 1024))
    for i in range(m):
        # 从文件名中解析出分类数字
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        # 分类标签
        hwLabels.append(classNumStr)
        # 训练数据矩阵
        trainingMat[i, :] = img2vector('digits/trainingDigits/%s' % fileNameStr)
    # 测试数据
    testFileList = listdir('digits/testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('digits/testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print('the classfier came back with: %d, the real answer is: %d'
              % (classifierResult, classNumStr))
        if (classifierResult != classNumStr):
            errorCount += 1.0
    print('\nthe total number of errors is: %d' % errorCount)
    print('\nthe total error rate is: %f' % (errorCount/float(mTest)))

handwritingClassTest()




