import numpy as np


class LinearRegression(object):
    def __init__(self):
        self.stepSize = 0.001
        self.maxIter = 10000
        self.bias = 1.0
        self.countIter = 0
        self.loss = 1.0
        self.minError = 1.0
        self.theta = np.array([[1.0],
                               [1.0]])

    def createData(self):
        self.data = np.array([[1.0, 1.0],
                              [1.0, 2.1],
                              [2.0, 1.1],
                              [4.0, 3.0],
                              [5.0, 1.1],
                              [2.1, 5.0]])
        self.lable = np.array([[8.1],
                               [12.0],
                               [10.0],
                               [22.0],
                               [18.3],
                               [27.4]])
        return self.data, self.lable

    def trainGD(self):
        while self.countIter < self.maxIter and self.loss >= self.minError:
            self.loss = 0
            y_pre = np.dot(self.data, self.theta) + self.bias
            self.loss += np.sum(0.5 * (y_pre - self.lable)**2) / len(self.data)
            if self.loss < self.minError:
                break
            print("Iter{}".format(self.countIter), self.theta[0], self.theta[1])
            for i in range(len(self.data)):
                self.theta -= (self.stepSize * self.data[i].T * (y_pre[i] - self.lable[i])).reshape([2, 1])
            self.countIter += 1
        print("Iter{}".format(self.countIter), self.theta[0], self.theta[1])

    def trainSGD(self):
        while self.countIter < self.maxIter and self.loss >= self.minError:
            self.loss = 0
            y_pre = np.dot(self.data, self.theta) + self.bias
            self.loss += np.sum(0.5 * (y_pre - self.lable)**2) / len(self.data)
            if self.loss < self.minError:
                break
            print("Iter{}".format(self.countIter), self.theta[0], self.theta[1])
            i = np.random.randint(0, len(self.data))
            temp = (self.stepSize * (y_pre[i] - self.lable[i]) * self.data[i]).T
            self.theta -= temp.reshape([2, 1])
            self.countIter += 1
        print("Iter{}".format(self.countIter), self.theta[0], self.theta[1])

    def trainBGD(self):
        """
        batch：2
        """
        while self.countIter < self.maxIter and self.loss >= self.minError:
            self.loss = 0
            y_pre = np.dot(self.data, self.theta) + self.bias
            self.loss += np.sum(0.5 * (y_pre - self.lable)**2) / len(self.data)
            if self.loss < self.minError:
                break
            print("Iter{}".format(self.countIter), self.theta[0], self.theta[1])
            i = np.random.randint(0, len(self.data))
            j = (i + 1) % len(self.data)
            temp = (self.stepSize * (y_pre[i] - self.lable[i]) * self.data[i]).T
            self.theta -= temp.reshape([2, 1])
            temp = (self.stepSize * (y_pre[j] - self.lable[j]) * self.data[j]).T
            self.theta -= temp.reshape([2, 1])
            self.countIter += 1
        print("Iter{}".format(self.countIter), self.theta[0], self.theta[1])


s = LinearRegression()
s.createData()
s.trainSGD()


