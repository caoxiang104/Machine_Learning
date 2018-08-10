from numpy import *
import time
import matplotlib.pyplot as plt
import random
import pandas as pd


def calcKernelMatrix(train_x, kernelOption):
    """
    train_x:  所有样本
    kernelOption: 核函数
    return: 核函数矩阵
    """
    numSamples = train_x.shape[0]
    kernelMatrix = mat(zeros((numSamples, numSamples)))  # 转化为矩阵类型
    for i in range(numSamples):
        kernelMatrix[:, i] = calcKernelValue(train_x, train_x[i, :], kernelOption)
    return kernelMatrix


def calcKernelValue(matrix_x, sample_x, kernelOption):
    """
    matrix_x: 所有样本
    sample_x: 单个样本
    kernelOption: 核函数
    return: 单个样本的核值
    """
    kernelType = kernelOption[0]
    numSamples = matrix_x.shape[0]
    kernelValue = mat(zeros((numSamples, 1)))

    if kernelType == 'linear':  # 线性核
        kernelValue = matrix_x * sample_x.T
    elif kernelType == 'rbf':  # 高斯核
        sigma = kernelOption[1]
        if sigma == 0:
            sigma = 1.0
        for i in range(numSamples):
            diff = matrix_x[i, :] - sample_x
            kernelValue[i] = exp(diff*diff.T/(-2.0*sigma**2))
    else:
        raise NameError("核函数输入格式不对，请使用高斯核或者线性核函数！")
    return kernelValue


class SVM(object):
    def __init__(self, dataSet, labels, C, toler, kernelOption):
        """
        dataSet: 数据集
        labels: 类别标签
        C: 松弛变量
        toler: 迭代终止条件
        kernelOption: 核函数
        """
        self.train_x = dataSet
        self.train_y = labels
        self.C = C
        self.toler = toler
        self.numSample = dataSet.shape[0]  # 样例数目
        self.alphas = mat(zeros((self.numSample, 1)))  # 拉格朗日系数
        self.b = 0
        self.errorCache = mat(zeros((self.numSample, 2)))
        self.kernelOpt = kernelOption
        self.kernelMat = calcKernelMatrix(self.train_x, self.kernelOpt)

    def calcError(self, alpha_k):
        """
        alpha_k: 第k个拉格朗日系数
        return: 第k个样本的误差
        """
        output_k = float(multiply(self.alphas, self.train_y).T * self.kernelMat[:, alpha_k] + self.b)
        error_k = output_k - float(self.train_y[alpha_k])
        return error_k

    def updateError(self, alpha_k):
        """
        alpha_k: 第k个拉格朗日系数
        return: 第k个拉格朗日系数的误差缓存
        """
        error = self.calcError(alpha_k)
        self.errorCache[alpha_k] = [1, error]

    def selectAlpha_j(self, alpha_i, error_i):
        """
        alpha_i: 第i个拉格朗日系数
        error_i: 第i个拉格朗日系数所求误差
        return: 第k个拉格朗日系数和其的误差
        """
        self.errorCache[alpha_i] = [1, error_i]
        candidateAlphaList = nonzero(self.errorCache[:, 0].A)[0]  # 支持向量点，也就是非边界点
        maxStep = 0
        alpha_j = 0
        error_j = 0

        if len(candidateAlphaList) > 1:  # 在支持向量里面找最大步长点
            for alpha_k in candidateAlphaList:
                if alpha_k == alpha_i:
                    continue
                error_k = self.calcError(alpha_k)
                if abs(error_k - error_i) > maxStep:
                    maxStep = abs(error_k - error_i)
                    error_j = error_k
                    alpha_j = alpha_k
        else:
            # 随机选择alpha_j
            alpha_j = alpha_i
            while alpha_j == alpha_i:
                alpha_j = int(random.uniform(0, self.numSample))
            error_j = self.calcError(alpha_j)
        return alpha_j, error_j

    def innerLoop(self, alpha_i):
        """
        alpha_i:
        return:
        描述: 优化alpha_i和alpha_j
        """
        error_i = self.calcError(alpha_i)
        """
        检查并挑选出违反KKT条件的拉格朗日系数
        KKT条件：
        1) yi * f(i) >= 1 and alpha == 0 (边界点)
	    2) yi * f(i) == 1 and 0 < alpha < C (支持向量)
	    3) yi * f(i) <= 1 and alpha == C (软间隔内点)
	    违反KKT条件
	    因为y[i] * E_i = y[i] * f(i) - y[i]^2 = y[i] * f(i) - 1，所以：
	    1) if y[i] * E_i < 0, so yi * f(i) < 1, if alpha < C, violate!(alpha = C will be correct) 
	    2) if y[i] * E_i > 0, so yi * f(i) > 1, if alpha > 0, violate!(alpha = 0 will be correct)
	    3) if y[i] * E_i == 0, so yi * f(i) == 1, it is on the boundary, needless optimized
        """
        if (self.train_y[alpha_i] * error_i < -self.toler and self.alphas[alpha_i] < self.C) or \
                (self.train_y[alpha_i] * error_i > self.toler and self.alphas[alpha_i] > 0):
            # 第一步：选择alpha_j
            alpha_j, error_j = self.selectAlpha_j(alpha_i, error_i)
            alpha_i_old = self.alphas[alpha_i].copy()
            alpha_j_old = self.alphas[alpha_j].copy()

            # 第二步：对alpha_j计算下界L和上界H
            if self.train_y[alpha_i] != self.train_y[alpha_j]:
                L = max(0, self.alphas[alpha_j] - self.alphas[alpha_i])
                H = min(self.C, self.C + self.alphas[alpha_j] - self.alphas[alpha_i])
            else:
                L = max(0, self.alphas[alpha_i] + self.alphas[alpha_j] - self.C)
                H = min(self.C, self.alphas[alpha_j] + self.alphas[alpha_i])
            if L == H:
                return 0

            # 第三步：计算eta,样本i与j的相似程度
            eta = 2.0 * self.kernelMat[alpha_i, alpha_j] - self.kernelMat[alpha_i, alpha_j] -\
                self.kernelMat[alpha_j, alpha_j]
            if eta >= 0:
                return 0

            # 第四步：更新alpha_j
            self.alphas[alpha_j] -= self.train_y[alpha_j] * (error_i - error_j) / eta

            # 第五步：修剪alpha_j
            if self.alphas[alpha_j] > H:
                self.alphas[alpha_j] = H
            if self.alphas[alpha_j] < L:
                self.alphas[alpha_j] = L

            # 第六步：如果alpha_j 几乎没变化，返回
            if abs(alpha_j_old - self.alphas[alpha_j]) < 0.00001:
                self.updateError(alpha_j)
                return 0

            # 第七步：优化alpha_j后更新alpha_i
            self.alphas[alpha_i] += self.train_y[alpha_i] * self.train_y[alpha_j] * \
                                    (alpha_j_old - self.alphas[alpha_j])

            # 第八步：更新偏置b
            b1 = self.b - error_i - self.train_y[alpha_i] * (self.alphas[alpha_i] - alpha_i_old) - \
                self.train_y[alpha_j] * (self.alphas[alpha_j] - alpha_j_old)
            b2 = self.b - error_j - self.train_y[alpha_i] * (self.alphas[alpha_i] - alpha_i_old) - \
                self.train_y[alpha_j] * (self.alphas[alpha_j] - alpha_j_old)
            if self.C > self.alphas[alpha_i] > 0:
                self.b = b1
            elif self.C > self.alphas[alpha_j] > 0:
                self.b = b2
            else:
                self.b = (b1 + b2) / 2.0

            # 第九步：更新alpha_i和alpha_j的误差缓存
            self.updateError(alpha_i)
            self.updateError(alpha_j)
            return 1
        else:
            return 0

    def trainSVM(self, maxIter):
        """
        train_x: 数据集
        train_y: 类别标签
        C: 松弛变量
        toler: 迭代终止条件
        maxIter: 最大迭代数
        kernelOption: 核函数
        """
        # 计算训练时间
        startTime = time.time()

        # 开始训练
        enterSet = True
        alphaPairsChanged = 0
        iterCount = 0

        """
        迭代终止条件：
        1）达到最大迭代步数
        2）遍历所以样本时没有alpha改变，即所有拉格朗日系数满足KKT约束
        """
        while (iterCount < maxIter) and ((alphaPairsChanged > 0) or enterSet):
            alphaPairsChanged = 0

            # 对所有训练样本更新拉格朗如系数
            if enterSet:
                for i in range(self.numSample):
                    alphaPairsChanged += self.innerLoop(i)
                print("--迭代：%d 整个数据集，alpha改变：%d" %(iterCount, alphaPairsChanged))
                iterCount += 1
            # 更新alpha不为0和C的值，即支持向量点
            else:
                nonBoundAlphaList = nonzero((self.alphas.A > 0) * (self.alphas.A < self.C))[0]
                for i in nonBoundAlphaList:
                    alphaPairsChanged += self.innerLoop(i)
                print("--迭代：%d 非边界点，alpha改变：%d" %(iterCount, alphaPairsChanged))
                iterCount += 1

            # 循环遍历所有样本和非边界样本
            if enterSet:
                enterSet = False
            elif alphaPairsChanged == 0:
                enterSet = True
        print("训练完成!耗时：%fs!" %(time.time() - startTime))

    def testSVM(self, test_x, test_y):
        """
        test_x: 测试数据集
        test_y: 测试数据集标签
        return: 正确率
        """
        numTestSamples = test_x.shape[0]
        supportVectorIndex = nonzero(self.alphas.A > 0)[0]
        supportVectors = self.train_x[supportVectorIndex]
        supportVectorLabels = self.train_y[supportVectorIndex]
        supportVectorAlphas = self.alphas[supportVectorIndex]
        matchCount = 0
        for i in range(numTestSamples):
            kernelValue = calcKernelValue(supportVectors, test_x[i, :], self.kernelOpt)
            predict = kernelValue.T * multiply(supportVectorLabels, supportVectorAlphas) + self.b
            if sign(predict) == sign(test_y[i]):
                matchCount += 1
        accuracy = float(matchCount) / numTestSamples
        return accuracy

    def showSVM(self):
        if self.train_x.shape[1] != 2:
            print("抱歉，不能画一维数据！")
            return 1

        # 画样本
        for i in range(self.numSample):
            if self.train_y[i] == -1:
                plt.plot(self.train_x[i, 0], self.train_x[i, 1], 'or')
            elif self.train_y[i] == 1:
                plt.plot(self.train_x[i, 0], self.train_x[i, 1], 'ob')

        # 描绘支持向量
        supportVectorsIndex = nonzero(self.alphas.A > 0)[0]
        for i in supportVectorsIndex:
            plt.plot(self.train_x[i, 0], self.train_x[i, 1], 'oy')

        # 画分界线
        w = zeros((2, 1))
        for i in supportVectorsIndex:
            w += multiply(self.alphas[i] * self.train_y[i], self.train_x[i, :].T)
        x1 = min(self.train_x[:, 0])[0, 0]
        x2 = max(self.train_x[:, 0])[0, 0]
        y1 = float(-self.b - w[0] * x1) / w[1]
        y2 = float(-self.b - w[0] * x2) / w[1]
        plt.plot([x1, x2], [y1, y2], '-g')
        plt.show()


def main():
    df = pd.read_csv('svm.csv')
    train_x = df.values[:80, :2].tolist()
    train_y = df.values[:80, 2:].tolist()
    test_x = df.values[80:, :2].tolist()
    test_y = df.values[80:, 2:].tolist()
    C = 0.6
    toler = 0.0001
    maxIter = 50
    train_x = mat(train_x)
    train_y = mat(train_y)
    test_x = mat(test_x)
    test_y = mat(test_y)
    svm = SVM(train_x, train_y, C, toler, kernelOption=('linear', 1.0))
    svm.trainSVM(maxIter)
    accuracy = svm.testSVM(test_x, test_y)
    print(accuracy)
    svm.showSVM()


if __name__ == '__main__':
    main()