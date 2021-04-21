#! /usr/bin/env python
# -*- coding:utf-8 -*-
"""
author: Vesper Huang
"""
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("../../../data/MNIST_data/", one_hot=True)

print("Training data size: ", mnist.train.num_examples)
print("Validating data size: ", mnist.validation.num_examples)
print("Testing data size: ", mnist.test.num_examples)

print("Example training data: ", mnist.train.images[0] )
print("Example training data label: ", mnist.train.labels[0])

batch_size = 100
xs, ys = mnist.train.next_batch(batch_size)    # 从train的集合中选取batch_size个训练数据。
print("X shape:", xs.shape)
print("Y shape:", ys.shape)