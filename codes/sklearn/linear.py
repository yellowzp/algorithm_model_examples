#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
我们定义向量 w = (w_1,..., w_p) 作为 coef_ ，定义 w_0 作为 intercept_
"""

from util import Util
from sklearn import linear_model
from sklearn import datasets
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score


def linear():
    """ 普通最小二乘法

    该方法使用 X 的奇异值分解来计算最小二乘解。如果 X 是一个 size 为 (n, p) 的矩阵，设 n >= p ，则该方法的复杂度为 O(n p^2)
    :return:
    """
    reg = linear_model.LinearRegression()
    diabetes = datasets.load_diabetes()

    # Use only one feature
    diabetes_X = diabetes.data[:, np.newaxis, 2]

    diabetes_X_train = diabetes_X[:-20]
    diabetes_X_test = diabetes_X[-20:]
    diabetes_y_train = diabetes.target[:-20]
    diabetes_y_test = diabetes.target[-20:]

    reg.fit(diabetes_X_train, diabetes_y_train)
    diabetes_y_pre = reg.predict(diabetes_X_test)

    print('Coefficients: \n', reg.coef_)

    # The mean squared error
    print("Mean squared error: %.2f" % mean_squared_error(diabetes_y_test, diabetes_y_pre))

    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % r2_score(diabetes_y_test, diabetes_y_pre))

    Util.plot([diabetes_X_train, diabetes_y_train], [[diabetes_X_test, diabetes_y_pre]])


def ridge_regression():
    """岭回归

    这种方法与 普通最小二乘法 的复杂度是相同的
    :return:
    """
    reg = linear_model.Ridge(alpha=.5)
    reg.fit([[0, 0], [0, 0], [1, 1]], [0, .1, 1])
    print reg.coef_
    print reg.intercept_

    # 设置正则化参数：广义交叉验证
    # 通过内置的 Alpha 参数的交叉验证来实现岭回归
    reg = linear_model.RidgeCV(alphas=[0.1, 1.0, 10.0])
    reg.fit([[0, 0], [0, 0], [1, 1]], [0, .1, 1])
    print reg.alpha_


def lasso():
    """
    是估计稀疏系数的线性模型
    使用了 coordinate descent （坐标下降算法）来拟合系数
    Lasso 回归产生稀疏模型，因此可以用于执行特征选择
    :return:
    """
    reg = linear_model.Lasso(alpha=0.1)
    reg.fit([[0, 0], [1, 1]], [0, 1])
    print reg.predict([[1, 1]])

    #


if __name__ == "__main__":
    # linear()
    # ridge_regression()
    lasso()
