import numpy as np
from scipy import stats
"""
http://www.hankcs.com/ml/em-algorithm-and-its-generalization.html
"""


class EM(object):
    def __init__(self, source=None):
        self.dataSets = source

    def createData(self):
        """
        构造数据集
        """
        self.dataSets = np.array([[1, 0, 0, 0, 1, 1, 0, 1, 0, 1],
                         [1, 1, 1, 1, 0, 1, 1, 1, 1, 1],
                         [1, 0, 1, 1, 1, 1, 1, 0, 1, 1],
                         [1, 0, 1, 0, 0, 0, 1, 1, 0, 0],
                         [0, 1, 1, 1, 0, 1, 1, 1, 0, 1]])
        return self.dataSets

    def singleEM(self, dataSets, thetaA, thetaB):
        """
        输入：数据集，初始模型参数
        输出：更新后的模型参数
        描述：单次em计算
        """
        counts = {'A': {'H': 0, 'T': 0}, 'B': {'H': 0, 'T': 0}}
        for data in dataSets:
            lenData = len(data)
            numHeads = sum(data)
            numTails = lenData - numHeads
            contributionA = stats.binom.pmf(numHeads, lenData, thetaA)
            contributionB = stats.binom.pmf(numHeads, lenData, thetaB)
            weightA = contributionA / (contributionA + contributionB)
            weightB = contributionB / (contributionA + contributionB)
            counts['A']['H'] += weightA * numHeads
            counts['A']['T'] += weightA * numTails
            counts['B']['H'] += weightB * numHeads
            counts['B']['T'] += weightB * numTails
        newThetaA = counts['A']['H'] / (counts['A']['H'] + counts['A']['T'])
        newThetaB = counts['B']['H'] / (counts['B']['H'] + counts['B']['T'])
        return newThetaA, newThetaB

    def em(self, dataSets, thetaA, thetaB, tol=1e-6, iterations=10000):
        """
        输入：数据集，初始模型参数，迭代终止误差，迭代终止步数
        输出：最终模型参数和迭代步数
        """
        iteration = 0
        while iteration < iterations:
            newThetaA, newThetaB = self.singleEM(dataSets, thetaA, thetaB)
            deltaChange = np.abs(newThetaA - thetaA)
            if deltaChange < tol:
                break
            else:
                thetaA, thetaB = newThetaA, newThetaB
                iteration += 1
        return [newThetaA, newThetaB, iteration]


def main():
    e = EM()
    dataSets = e.createData()
    print(e.em(dataSets, 0.6, 0.5))


if __name__ == '__main__':
    main()