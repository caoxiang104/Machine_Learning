# -*- coding:utf-8 -*-
import numpy as np


class NaiveBayes(object):
    def __init__(self):
        pass

    def createDataSet(self):
        """
        描述：创建输入属性和输出类别
        输出：属性和类别
        """
        wordsList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                     ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                     ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                     ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                     ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                     ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
        classVec = [0, 1, 0, 1, 0, 1]  # 1表示侮辱性言论，0表示正常言论
        return wordsList, classVec

    def createVocaList(self, wordsList):
        """
        输入：样本词表
        输出：总词表
        """
        totalWords = set([])
        for words in wordsList:
            totalWords = totalWords | set(words)
        return list(totalWords)

    def setOfWordsToVec(self, wordsList, totalWords):
        """
        输入：样本词表，总词表
        输出: 词向量
        描述:构建词向量，采用词集模型
        """
        wordsVec = []
        for words in wordsList:
            tempWords = [0]*len(totalWords)
            for word in words:
                if word in totalWords:
                    tempWords[totalWords.index(word)] = 1
            wordsVec.append(tempWords)
        return wordsVec

    def bagOfWordsTovec(self, wordsList, totalWords):
        """
        输入：样本词表，总词表
        输出: 词向量
        描述:构建词向量，采用词袋模型
        """
        wordsVec = []
        for words in wordsList:
            tempWords = [0] * len(totalWords)
            for word in words:
                if word in totalWords:
                    tempWords[totalWords.index(word)] = +1
            wordsVec.append(tempWords)
        return wordsVec

    def trainNB(self, wordsVec, totalWords, classVec):
        """
        p(ci)好求，用样本集中，ci的数量/总样本数即可
        p(ω|ci)由于各个条件特征相互独立且地位相同，
        p(ω|ci)=p(w0|ci)p(w1|ci)p(w2|ci)......p(wN|ci)p(ω|ci)，可以分别求
        p(w0|ci),p(w1|ci),p(w2|ci),......,p(wN|ci)，从而得到p(ω|ci).
        p(ωk|ci)=ωk在ci中出现的次数/ci的数量(ci中总词数：词袋模型)
        输入：词集向量，总词表，类别表
        输出：p(ci),p(ci'),p(w|ci),p(w|ci')
        """
        insult = np.ones(len(totalWords))
        normal = np.ones(len(totalWords))
        pIns = (sum(classVec) + 1) / (len(classVec) + 2)  # p(ci)
        pNor = (len(classVec) - sum(classVec) + 1) / (len(classVec) + 2)  # p(ci')
        for i in range(len(classVec)):
            if classVec[i] == 1:
                insult += wordsVec[i]
            else:
                normal += wordsVec[i]
        pInsult = insult / (sum(classVec) + 2)
        pNormal = normal / (len(classVec) - sum(classVec) + 2)
        return pIns, pNor, pInsult, pNormal

    def classifyNB(self, testVec, pInsult, pNormal, pIns, pNor):
        """
        输入：p(ci),p(ci'),p(w|ci),p(w|ci'),测试属性
        输出：推测类别
        """
        sum1 = 1
        sum2 = 1
        for i in range(len(pInsult)):
            if testVec[i] == 1:
                sum1 *= pInsult[i]
                sum2 *= pNormal[i]
        if sum1 * pIns > pNor * sum2:
            return 1
        else:
            return 0

    def testNB(self):
        wordsList, classVec = self.createDataSet()
        totalWords = self.createVocaList(wordsList)
        wordsVec = self.setOfWordsToVec(wordsList, totalWords)
        pIns, pNor, pInsult, pNormal = self.trainNB(wordsVec, totalWords, classVec)
        testEntry = [['love', 'my', 'dalmation'],
                     ['stupid', 'garbage']]
        testVec = self.setOfWordsToVec(testEntry, totalWords)
        for i in range(len(testVec)):
            if self.classifyNB(testVec[i], pInsult, pNormal, pIns, pNor):
                print("分类为侮辱性词语:", testEntry[i])
            else:
                print("分类为正常词语:", testEntry[i])


def main():
    nb = NaiveBayes()
    nb.testNB()


if __name__ == '__main__':
    main()