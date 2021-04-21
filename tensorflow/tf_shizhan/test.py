#! /usr/bin/env python
# -*- coding:utf-8 -*-
"""
author: Vesper Huang
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

sess = tf.Session()

#
# bias = tf.Variable(tf.constant(0.1, shape=[5]))
# bias1 = tf.Variable([0.1, 0.1])
# print(bias1)

result = tf.equal([1, 0], [1, 1])
print(sess.run(result))