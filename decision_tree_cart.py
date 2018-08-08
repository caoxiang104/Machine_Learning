from math import log
import treePlotter
import operator
import pandas as pd


class DecisionTree(object):
    def __init__(self):
        self.dataSet = []
    
    def calcGini(self, dataSet):
        """
        输入：数据集
        输出：数据集的基尼指数
        描述：计算给定数据集的基尼指数；基尼指数越小，数据集的混乱程度越大
        """
        numEntries = len(dataSet)
        labelList = [example[-1] for example in dataSet]
        labelCounter = {}
        for label in labelList:
            if label not in labelCounter.keys():
                labelCounter[label] = 0
            labelCounter[label] += 1
        gini = 1.0
        for key in labelCounter.keys():
            prob = float(labelCounter[key]) / numEntries
            gini -= prob*prob
        return gini

    def splitDataSet(self, dataSet, axis, value):
        """
        输入：数据集，选择维度，选择值
        输出：划分数据集,分为两类
        描述：按照给定特征划分数据集；去除选择维度中等于选择值的项
        """
        retDataSet1 = []
        retDataSet2 = []
        for featureVec in dataSet:
            if featureVec[axis] == value:
                reducedFeatureVec = featureVec[:axis]
                reducedFeatureVec.extend(featureVec[axis + 1:])
                retDataSet1.append(reducedFeatureVec)
            else:
                reducedFeatureVec = featureVec[:axis]
                reducedFeatureVec.extend(featureVec[axis + 1:])
                retDataSet2.append(reducedFeatureVec)
        return retDataSet1, retDataSet2

    def spiltContinousDataSet(self, dataSet, axis, value, direction):
        """
        输入：数据集，选择维度，选择值，选择方向：左子树或者右子树
        输出：划分数据集,分为两类
        描述：按照给定特征划分数据集；去除选择维度中等于选择值的项
        """
        retDataSet = []
        for featureVec in dataSet:
            if direction == 0:
                if featureVec[axis] > value:
                    reducedFeatureVec = featureVec[:axis]
                    reducedFeatureVec.extend(featureVec[axis+1:])
                    retDataSet.append(reducedFeatureVec)
            else:
                if featureVec[axis] <= value:
                    reducedFeatureVec = featureVec[:axis]
                    reducedFeatureVec.extend(featureVec[axis + 1:])
                    retDataSet.append(reducedFeatureVec)
        return retDataSet

    def chooseBestFeatureToSplit(self, dataSet, labels):
        """
        输入：数据集
        输出：最好的划分维度
        描述：选择最好的数据集划分维度
        """
        numFeatures = len(dataSet[0]) - 1
        bestGiniIndex = 10000.0
        bestFeature = -1
        bestSplitDict = {}
        for i in range(numFeatures):
            featureList = [example[i] for example in dataSet]
            if type(featureList[0]).__name__ == 'float' or type(featureList[0]).__name__ == 'int':
                sortedFeatureList = sorted(featureList)
                splitList = []
                for j in range(len(sortedFeatureList) - 1):
                    splitList.append((sortedFeatureList[j] + sortedFeatureList[j + 1])/2.0)
                bestSplitGini = 10000.0
                splitLen = len(splitList)
                for j in range(splitLen):
                    value = splitList[j]
                    newGiniIndex = 0.0
                    subDataSet0 = self.spiltContinousDataSet(dataSet, i, value, 0)
                    subDataSet1 = self.spiltContinousDataSet(dataSet, i, value, 1)
                    prob0 = len(subDataSet0) / float(len(dataSet))
                    newGiniIndex += prob0 * self.calcGini(subDataSet0)
                    prob1 = len(subDataSet1) / float(len(dataSet))
                    newGiniIndex += prob1 * self.calcGini(subDataSet1)
                    if newGiniIndex < bestSplitGini:
                        bestSplitGini = newGiniIndex
                        bestSplitIndex = j
                bestSplitDict[labels[i]] = splitList[bestSplitIndex]
                GiniIndex = bestSplitGini
            else:
                bestSplitGini = 10000.0
                uniqueVals = set(featureList)
                for value in uniqueVals:
                    newGiniIndex = 0.0
                    subDataSet0, subDataSet1 = self.splitDataSet(dataSet, i, value)
                    prob0 = len(subDataSet0) / float(len(dataSet))
                    newGiniIndex += prob0 * self.calcGini(subDataSet0)
                    prob1 = len(subDataSet1) / float(len(dataSet))
                    newGiniIndex += prob1 * self.calcGini(subDataSet1)
                    if newGiniIndex < bestSplitGini:
                        bestSplitGini = newGiniIndex
                        bestSplitVal = value
                bestSplitDict[labels[i]] = bestSplitVal
                GiniIndex = bestSplitGini
            if GiniIndex < bestGiniIndex:
                bestGiniIndex = GiniIndex
                bestFeature = i
        bestSplitValue = bestSplitDict[labels[bestFeature]]
        if type(dataSet[0][bestFeature]).__name__ == 'float' or type(dataSet[0][bestFeature]).__name__ == 'int':
            labels[bestFeature] = labels[bestFeature] + '<=' + str(bestSplitValue)
            for i in range(len(dataSet[0])):
                if dataSet[i][bestFeature] <= bestSplitValue:
                    dataSet[i][bestFeature] = 1
                else:
                    dataSet[i][bestFeature] = 0
        return bestFeature, bestSplitValue

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
        if len(dataSet) == 0:
            return 0
        if len(dataSet[0]) == 1:  # 遍历完所有特征返回出现次数最多的
            return self.majorityCnt(dataSet)
        bestFeature, value = self.chooseBestFeatureToSplit(dataSet, labels)
        bestFeatureLabel = labels[bestFeature]
        myTree = {bestFeatureLabel: {}}
        del (labels[bestFeature])
        featureVals = [example[bestFeature] for example in dataSet]
        retDataSet1, retDataSet2 = self.splitDataSet(dataSet, bestFeature, value)
        subLabels = labels[:]
        if not subLabels:
            return self.majorityCnt(dataSet)
        if retDataSet1 is not None:
            myTree[bestFeatureLabel][value] = self.createTree(retDataSet1, subLabels)
        else:
            return
        if retDataSet2 is not None:
            myTree[bestFeatureLabel]["No" + value] = self.createTree(retDataSet2, subLabels)
        else:
            return
        return myTree


def main():
    df = pd.read_csv('watermelon_4_2.csv')
    data = df.values[:, 1:].tolist()
    data_full = data[:]
    labels = df.columns.values[1:-1].tolist()
    labels_full = labels[:]
    d = DecisionTree()
    myTree = d.createTree(data, labels)
    treePlotter.createPlot(myTree)


if __name__=='__main__':
    main()
