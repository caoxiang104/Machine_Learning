# -*- coding:utf-8 -*-
"""
决策树：ID3算法实例
ID3 使用信息增益作为选择特征的准则
信息增益 = 划分前熵 - 划分后熵。 信息增益越大，则意味着使用属性 a 来进行划分所获得的
“纯度提升”越大。也就是说，用属性 a 来划分训练集，得到的结果中纯度比较高。
ID3 仅仅适用于二分类问题。 ID3 仅仅能够处理离散属性。
"""
from math import log
import operator


class DecisionTree(object):
    def __init__(self):
        self.dataSet = []
        self.labels = []

    def createdataSet(self):
        """
        输出：数据集和特征 
        """
        self.dataSet = [['长', '粗', '男'],   # 类别：男和女
                        ['短', '粗', '男'],
                        ['短', '粗', '男'],
                        ['长', '细', '女'],
                        ['短', '细', '女'],
                        ['短', '粗', '女'],
                        ['长', '粗', '女'],
                        ['长', '粗', '女']]
        # self.dataSet = [['男'],  # 类别：男和女
        #                 ['男'],
        #                 ['男'],
        #                 ['女'],
        #                 ['女'],
        #                 ['女'],
        #                 ['女'],
        #                 ['女']]
        self.labels = ['头发', '声音']  # 两个特征
        return self.dataSet, self.labels

    def calShannonEntropy(self, dataSet):
        """
        输入：数据集
        输出：数据集的香农熵
        描述：计算给定数据集的香农熵；熵越大，数据集的混乱程度越大
        """
        num = len(dataSet)  # 数据长度
        labelCount = {}  # 两个类别的数量统计
        for feature in dataSet:
            currentLabel = feature[-1]  # '男'或者'女'
            if currentLabel not in labelCount.keys():
                labelCount[currentLabel] = 0
            labelCount[currentLabel] += 1   # 统计每个类别数量
        shannonEntropy = 0
        for key in labelCount.keys():
            temp = float(labelCount[key])/num  # 计算单个类的熵值
            shannonEntropy -= temp*log(temp, 2)
        return shannonEntropy

    def splitDataSet(self, dataSet, axis, value):
        """
        输入：数据集，选择维度，选择值
        输出：划分数据集
        描述：按照给定特征划分数据集；去除选择维度中等于选择值的项
        """
        retDataSet = []
        for featureVec in dataSet:
            if featureVec[axis] == value:
                reducedFeatureVec = featureVec[:axis]
                reducedFeatureVec.extend(featureVec[axis+1:])
                retDataSet.append(reducedFeatureVec)
        return retDataSet

    def majorityCount(self, classList):
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

    def chooseBestFeatureToSpilt(self, dataSet):
        """
        输入：数据集
        输出：最好的划分维度
        描述：选择最好的数据集划分维度
        """
        numFeatures = len(dataSet[0]) - 1  # 除了类别，都是特征
        baseEntropy = self.calShannonEntropy(dataSet)   # 初始熵
        bestInfoGain = 0  # 最佳信息增益
        bestFeature = -1  # 最佳特征位置
        for i in range(numFeatures):
            featureList = [example[i] for example in dataSet]
            uniqueValues = set(featureList)
            newEntropy = 0
            for value in uniqueValues:
                subDataSet = self.splitDataSet(dataSet, i, value)   # 按特征分类的子集
                prob = len(subDataSet) / float(len(dataSet))
                newEntropy += prob*self.calShannonEntropy(subDataSet)  # 按特征分类的熵
            infoGain = baseEntropy - newEntropy  # 信息增益
            if infoGain > bestInfoGain:
                bestInfoGain = infoGain
                bestFeature = i
        return bestFeature

    def createTree(self, dataSet, labels):
        """
        输入：数据集，特征标签
        输出：决策树
        描述：递归构建决策树，利用上述的函数
        """
        classList = [example[-1] for example in dataSet]  # 类别: 男或者女
        if classList.count(classList[0]) == len(classList):
            # 分割出的所有样本属于同一类  停止递归
            return classList[0]
        if len(dataSet[0]) == 1:  # 数据只有类别，输出最多的类别
            # 由于每次分割消耗一个feature，当没有feature的时候停止递归，返回当前样本集中大多数sample的label
            return self.majorityCount(classList)
        bestFeature = self.chooseBestFeatureToSpilt(dataSet)  # 选择最优特征
        bestFeatureLabel = labels[bestFeature]
        myTree = {bestFeatureLabel:{}}   # 分类结果以字典形式保留
        del(labels[bestFeature])    # 删除该特征标志
        featureValues = [example[bestFeature] for example in dataSet]
        unqiueValues = set(featureValues)
        for value in unqiueValues:  # 选择余下特征
            subLabels = labels[:]
            myTree[bestFeatureLabel][value] = self.createTree(
                self.splitDataSet(dataSet, bestFeature, value), subLabels)
        return myTree


def main():
    D = DecisionTree()
    dataSet, labels = D.createdataSet()  # 创造示列数据
    print(D.createTree(dataSet, labels))  # 输出决策树模型结果


if __name__=='__main__':
    main()