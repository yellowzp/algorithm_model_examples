#! /usr/bin/env python
# -*- coding: utf-8 -*-

from sklearn.linear_model import LogisticRegression
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pdb
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


class MyLogisticRegression(object):

    @classmethod
    def train(cls):
        """
        C是正则项的参数

        :return:
        """
        iris = datasets.load_iris()  # iris数据集
        x = iris.data[:, [2, 3]]  # 只拿第3第4个特征做为训练特征
        y = iris.target
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)  # 交叉验证
        sc = StandardScaler()
        sc.fit(x_train)  # 获取特征的均值，标准差
        print sc.mean_  # 均值
        print sc.scale_  # 标准差
        x_train_std = sc.transform(x_train)  # 归一化
        x_test_std = sc.transform(x_test)
        lr = LogisticRegression(C=1000, penalty='l2', random_state=0)
        lr.fit(x_train_std, y_train)
        y_pre = lr.predict(x_test_std)  # 查看第一个测试样本属于各个类别的概率
        print classification_report(y_test, y_pre)  # 准确率，召回率，F1
        print lr.score(x_test_std, y_test)  # 测试集平均精确率


if __name__ == "__main__":
    MyLogisticRegression.train()