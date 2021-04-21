#! /usr/bin/env python
# -*- coding:utf-8 -*-
"""
author: Vesper Huang
"""

import tensorflow as tf

A = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

with tf.Session() as sess:
    print(sess.run(tf.nn.softmax(A)))
    print(sess.run(tf.nn.softmax([0, 0, 1.0])))

