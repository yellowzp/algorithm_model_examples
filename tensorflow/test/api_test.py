#! /usr/bin/env python
# -*- coding:utf-8 -*-
"""
@author: Vesper Huang
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


def test_tile():
    """
    tf.tile: 张量自己多份拼接
    :return:
    """
    tensor = tf.constant([1, 2], dtype=tf.int64, name="tensor")
    # print("tensor: %s" % tensor)
    new_tensor = tf.tile(tensor, [2], name="new_tensor")
    # print("new_tensor: %s" % new_tensor)
    with tf.Session() as sess:
        print("tensor", sess.run(tensor))
        print("new_tensor", sess.run(new_tensor))


def test_where():
    """
    tf.where(condition, x=None, y=None, name=None)
    返回值是对应元素，condition中元素为True的元素替换为x中的元素，为False的元素替换为y中对应元素
    :return:
    """
    x = [[1, 2, 3], [4, 5, 6]]
    y = [[7, 8, 9], [10, 11, 12]]
    condition3 = [[True, False, False],
                  [False, True, True]]
    condition4 = [[True, False, False],
                  [True, True, False]]
    with tf.Session() as sess:
        print(sess.run(tf.where(condition3, x, y)))
        print(sess.run(tf.where(condition4, x, y)))


if __name__ == "__main__":
    print("start")
    # test_tile()
    test_where()