# -*- coding:utf-8 -*-
"""
决策树：C4.5算法实例
C4.5 使用信息增益比作为选择特征的准则
信息增益比 = 信息增益 /划分前熵 选择信息增益比最大的作为最优特征
C4.5 处理连续特征是先将特征取值排序，以连续两个值中间值作为划分标准。尝试每一种划分，
并计算修正后的信息增益，选择信息增益最大的分裂点作为该属性的分裂点
"""
from math import log
import operator
import treePlotter


class DecisionTree(object):
    def __init__(self):
        self.dataSet = []
        self.testSet = []
        self.labels = []

    def calShannonEntropy(self, dataSet):
        """
        输入：数据集
        输出：数据集的香农熵
        描述：计算给定数据集的香农熵；熵越大，数据集的混乱程度越大
        """
        numEntries = len(dataSet)  # 数据大小
        labelCounts = {}
        for featureVector in dataSet:  # 统计每个类别的数量
            currentLabel = featureVector[-1]
            if currentLabel not in labelCounts.keys():
                labelCounts[currentLabel] = 0
            labelCounts[currentLabel] += 1
        shannonEntropy = 0.0
        for key in labelCounts:  # 计算香农熵
            prob = float(labelCounts[key]) / numEntries
            shannonEntropy -= prob * log(prob, 2)
        return shannonEntropy

    def splitDataSet(self, dataSet, axis, value):
        """
        输入：数据集，选择维度，选择值
        输出：划分数据集
        描述：按照给定特征划分数据集；去除选择维度中等于选择值的项
        """
        retDataSet = []
        numFeatures = len(dataSet[0]) - 1
        for featureVec in dataSet:
            if featureVec[axis] == value:
                reduceFeatureVec = featureVec[:axis]
                reduceFeatureVec.extend(featureVec[axis+1:])
                retDataSet.append(reduceFeatureVec)
        return retDataSet

    def chooseBestFeatureToSplit(self, dataSet):
        """
        输入：数据集
        输出：最好的划分维度
        描述：选择最好的数据集划分维度
        """
        numFeatures = len(dataSet[0]) - 1
        baseEntroy = self.calShannonEntropy(dataSet)  # 初始熵
        bestInfoGainRatio = 0.0  # 初始信息增益比
        bestFeature = -1  # 初始最佳特征位置
        for i in range(numFeatures):
            featureList = [example[i] for example in dataSet]
            unqiyeVals = set(featureList)  # 当前特征包含信息
            newEntropy = 0.0  # 初始新熵
            splitInfo = 0.0 # 初始分割增益
            for value in unqiyeVals:
                subDataSet = self.splitDataSet(dataSet, i, value)
                prob = len(subDataSet)/ float(len(dataSet))
                newEntropy += prob * self.calShannonEntropy(subDataSet)
                splitInfo -= prob * log(prob, 2)
            infoGain = baseEntroy - newEntropy
            if splitInfo == 0:  # 修复过拟合bug
                continue
            infoGainRatio = infoGain / splitInfo
            if infoGainRatio > bestInfoGainRatio:
                bestInfoGainRatio = infoGainRatio
                bestFeature = i
            return bestFeature

    def majorityCnt(self, classList):
        """
        输入：分类类别列表
        输出：子节点的分类
        描述：数据集已经处理了所有属性，但是类标签依然不是唯一的，
          采用多数判决的方法决定该子节点的分类
        """
        classCount = {}
        for vote in classList:
            if vote not in classCount.keys():
                classCount[vote] = 0
            classCount[vote] += 1
        sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
        return sortedClassCount[0][0]

    def createTree(self, dataSet, labels):
        """
        输入：数据集，特征标签
        输出：决策树
        描述：递归构建决策树，利用上述的函数
        """
        classList = [example[-1] for example in dataSet]
        if classList.count(classList[0]) == len(classList):  # 类别完全相同，停止划分
            return classList[0]
        if len(dataSet[0]) == 1:  # 遍历完所有特征返回出现次数最多的
            return self.majorityCnt(dataSet)
        bestFeature = self.chooseBestFeatureToSplit(dataSet)
        bestFeatureLabel = labels[bestFeature]
        myTree = {bestFeatureLabel:{}}
        del(labels[bestFeature])
        featureVals = [example[bestFeature] for example in dataSet]
        uniqueVals = set(featureVals)
        for value in uniqueVals:
            subLabels = labels[:]
            myTree[bestFeatureLabel][value] = self.createTree(
                self.splitDataSet(dataSet, bestFeature, value), subLabels)
        return myTree

    def classify(self, inputTree, featureLabel, testVec):
        """
        输入：决策树，分类标签，测试数据
        输出：决策结果
        描述：跑决策树
        """
        firstStr = list(inputTree.keys())[0]
        secondDict = inputTree[firstStr]
        featureIndex = featureLabel.index(firstStr)
        for key in secondDict.keys():
            if testVec[featureIndex] == key:
                if type(secondDict[key]).__name__ == 'dict':
                    classLabel = self.classify(secondDict[key], featureLabel, testVec)
                else:
                    classLabel = secondDict[key]
        return classLabel

    def classifyAll(self, inputTree, featureLabels, testDataSet):
        """
        输入：决策树，分类标签，测试数据集
        输出：决策结果
        描述：跑决策树
        """
        classLabelAll = []
        for testVec in testDataSet:
            classLabelAll.append(self.classify(inputTree, featureLabels, testVec))
        return classLabelAll

    def storeTree(self, inputTree, filename):
        """
        输入：决策树，保存文件路径
        输出：
        描述：保存决策树到文件
        """
        import pickle
        fw = open(filename, 'wb')
        pickle.dump(inputTree, fw)
        fw.close()

    def grabTree(self, filename):
        """
        输入：文件路径名
        输出：决策树
        描述：从文件读取决策树
        """
        import pickle
        fr = open(filename, 'rb')
        return pickle.load(fr)

    def createDataSet(self):
        """
        outlook->  0: sunny | 1: overcast | 2: rain
        temperature-> 0: hot | 1: mild | 2: cool
        humidity-> 0: high | 1: normal
        windy-> 0: false | 1: true

        """
        self.dataSet = [[0, 0, 0, 0, 'N'],
                   [0, 0, 0, 1, 'N'],
                   [1, 0, 0, 0, 'Y'],
                   [2, 1, 0, 0, 'Y'],
                   [2, 2, 1, 0, 'Y'],
                   [2, 2, 1, 1, 'N'],
                   [1, 2, 1, 1, 'Y']]
        self.labels = ['outlook', 'temperature', 'humidity', 'windy']
        return self.dataSet, self.labels

    def createTestSet(self):
        """
        outlook->  0: sunny | 1: overcast | 2: rain
        temperature-> 0: hot | 1: mild | 2: cool
        humidity-> 0: high | 1: normal
        windy-> 0: false | 1: true
        """
        self.testSet = [[0, 1, 0, 0],
                       [0, 2, 1, 0],
                       [2, 1, 1, 0],
                       [0, 1, 1, 1],
                       [1, 1, 0, 1],
                       [1, 0, 1, 0],
                       [2, 1, 0, 1]]
        return self.testSet


def main():
    D = DecisionTree()
    dataSet, labels = D.createDataSet()
    labels_tmp = labels[:]  # 拷贝，createTree会改变labels
    desicionTree = D.createTree(dataSet, labels_tmp)
    # storeTree(desicionTree, 'classifierStorage.txt')
    # desicionTree = grabTree('classifierStorage.txt')
    print('desicionTree:\n', desicionTree)
    treePlotter.createPlot(desicionTree)
    testSet = D.createTestSet()
    print('classifyResult:\n', D.classifyAll(desicionTree, labels, testSet))


if __name__=='__main__':
    main()