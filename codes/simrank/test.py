#! /usr/bin/env python
# -*- coding:utf-8 -*-
"""
author: Vesper Huang
"""

import numpy as np

# a = np.array([[1, 2], [3, 4]], dtype=float)
#
# print a
# print np.diag(np.diag(a))

# for idx in xrange(a.shape[1]):
#     print a[idx]
#     total = np.sum(a[idx])
#     a[idx] = a[idx] / total
#
# print a

# P = np.zeros((3, 3))
# P[0, 2] = 0.5
# P[1, 2] = 0.5
# P[2, 0] = 1
# P[2, 1] = 1
# P_t = P.transpose()
# print P
# print P_t
# S = np.identity(3)
# print S
# tmp = S
# for idx in xrange(0, 5):
#     new = np.dot(np.dot(P_t, tmp), P)
#     tmp = new
#     print tmp

a = np.array([1, 1, 1, 1], dtype=float)
print np.var(a)