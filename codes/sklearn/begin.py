#! /usr/bin/env python
# -*- coding: utf-8 -*-

import sklearn
import pickle
import numpy as np
from sklearn.externals import joblib

from sklearn import datasets
from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MultiLabelBinarizer

digits = datasets.load_digits()
clf = svm.SVC(gamma=0.001, C=100.)
clf.fit(digits.data[:-1], digits.target[:-1])
print clf.predict(digits.data[-1:])
print digits.target[-1:]

# 持久化，字符串保存
s = pickle.dumps(clf)

# 持久化，硬盘存储
joblib.dump(clf, 'filename.pkl')
clf = joblib.load('filename.pkl')
print clf.predict(digits.data[-1:])

# 再次训练和更新参数
rng = np.random.RandomState(0)
X = rng.rand(100, 10)
y = rng.binomial(1, 0.5, 100)
X_test = rng.rand(5, 10)

clf = svm.SVC()
clf.set_params(kernel='linear').fit(X, y)
print clf.predict(X_test)
clf.set_params(kernel='rbf').fit(X, y)
print clf.predict(X_test)

# 多分类与多标签拟合
X = [[1, 2], [2, 4], [4, 5], [3, 2], [3, 1]]
y = [0, 0, 1, 1, 2]
classif = OneVsRestClassifier(estimator=svm.SVC(random_state=0))
print classif.fit(X, y).predict(X)

## 标签二值化后
y = LabelBinarizer().fit_transform(y)
print classif.fit(X, y).predict(X)

## 多标签输出
y = [[0, 1], [0, 2], [1, 3], [0, 2, 3], [2, 4]]
y = MultiLabelBinarizer().fit_transform(y)
print classif.fit(X, y).predict(X)






