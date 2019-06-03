#! /usr/bin/env python
# -*- coding:utf-8 -*-
"""
author: Vesper Huang
"""

import numpy as np

a = np.array([[1, 2], [3, 4]], dtype=float)

print a
print np.diag(np.diag(a))

# for idx in xrange(a.shape[1]):
#     print a[:, idx]
#     total = np.sum(a[:, idx])
#     a[:, idx] = a[:, idx] / total
#
# print a