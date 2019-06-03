#! /usr/bin/env python
# -*- coding:utf-8 -*-
"""
author: Vesper Huang
"""

import tensorflow as tf
import numpy as np

rdm = np.random.RandomState(1)
dataset_size = 128
X = rdm.rand(dataset_size, 2)
print(X)
Y = [[int(x1+x2) < 1] for (x1, x2) in X]
print(Y)
