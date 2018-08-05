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
    def create_dataset(self):
        self.dataset = [['长', '粗', '男'],   # 类别：男和女
                        ['短', '粗', '男'],
                        ['短', '粗', '男'],
                        ['长', '细', '女'],
                        ['短', '细', '女'],
                        ['短', '粗', '女'],
                        ['长', '粗', '女'],
                        ['长', '粗', '女']]
        # self.dataset = [['男'],  # 类别：男和女
        #                 ['男'],
        #                 ['男'],
        #                 ['女'],
        #                 ['女'],
        #                 ['女'],
        #                 ['女'],
        #                 ['女']]
        self.labels = ['头发', '声音']  # 两个特征
        return self.dataset, self.labels

    def cal_shannon_entropy(self, dataset):
        """计算香农熵 -(p1 * log p1 + p2 * log p2 +　．．．　+ pn * log pn)"""
        num = len(dataset)  # 数据长度
        label_count = {}  # 两个类别的数量统计
        for feature in dataset:
            current_label = feature[-1]  # '男'或者'女'
            if current_label not in label_count.keys():
                label_count[current_label] = 0
            label_count[current_label] += 1   # 统计每个类别数量
        shannon_entropy = 0
        for key in label_count.keys():
            temp = float(label_count[key])/num  # 计算单个类的熵值
            shannon_entropy -= temp*log(temp, 2)
        return shannon_entropy

    def split_dataset(self, dataset, axis, value):
        """按某个特征分类数据"""
        ret_dataset = []
        for feature_vec in dataset:
            if feature_vec[axis] == value:
                reduced_feature_vec = feature_vec[:axis]
                reduced_feature_vec.extend(feature_vec[axis+1:])
                ret_dataset.append(reduced_feature_vec)
        return ret_dataset

    def majority_count(self, class_list):
        """按分类后类别数量排序，比如：最后分类为2男1女，则判定为男"""
        class_count = {}
        for vote in class_list:
            if vote not in class_count.keys():
                class_count[vote] = 0
            class_count[vote] += 1
        sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
        return sorted_class_count[0][0]

    def choose_best_feature_to_spilt(self, dataset):
        num_features = len(dataset[0]) - 1  # 除了类别，都是特征
        base_entropy = self.cal_shannon_entropy(dataset)   # 初始熵
        best_info_gain = 0  # 最佳信息增益
        best_feature = -1  # 最佳特征位置
        for i in range(num_features):
            feature_list = [example[i] for example in dataset]
            unique_values = set(feature_list)
            new_entropy = 0
            for value in unique_values:
                sub_dataset = self.split_dataset(dataset, i, value)   # 按特征分类的子集
                prob = len(sub_dataset) / float(len(dataset))
                new_entropy += prob*self.cal_shannon_entropy(sub_dataset)  # 按特征分类的熵
            info_gain = base_entropy - new_entropy  # 信息增益
            if info_gain > best_info_gain:
                best_info_gain = info_gain
                best_feature = i
        return best_feature

    def create_tree(self, dataset, labels):
        """创建决策树"""
        class_list = [example[-1] for example in dataset]  # 类别: 男或者女
        if class_list.count(class_list[0]) == len(class_list):  # 所有数据为同一类
            return class_list[0]
        if len(dataset[0]) == 1:  # 数据只有类别，输出最多的类别
            return self.majority_count(class_list)
        best_feature = self.choose_best_feature_to_spilt(dataset)  # 选择最优特征
        best_feature_label = labels[best_feature]
        my_tree = {best_feature_label:{}}   # 分类结果以字典形式保留
        del(labels[best_feature])    # 删除该特征标志
        feature_values = [example[best_feature] for example in dataset]
        unqiue_values = set(feature_values)
        for value in unqiue_values:  # 选择余下特征
            sub_labels = labels[:]
            my_tree[best_feature_label][value] = self.create_tree(
                self.split_dataset(dataset, best_feature, value), sub_labels)
        return my_tree


def main():
    man_or_woman = DecisionTree()
    dataSet, labels = man_or_woman.create_dataset()  # 创造示列数据
    print(man_or_woman.create_tree(dataSet, labels))  # 输出决策树模型结果


if __name__=='__main__':
    main()