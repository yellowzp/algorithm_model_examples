#! /usr/bin/env python
# -*- coding:utf-8 -*-
"""
@author: Vesper Huang
"""
class CQueue(object):

    def __init__(self):
        self.array = []


    def appendTail(self, value):
        """
        :type value: int
        :rtype: None
        """
        self.array += [value]


    def deleteHead(self):
        """
        :rtype: int
        """
        if not self.array:
            return -1
        tmp = self.array[0]
        self.array = self.array[1:]
        return tmp
