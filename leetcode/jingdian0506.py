#! /usr/bin/env python
# -*- coding:utf-8 -*-
"""
@author: Vesper Huang
"""
class Solution(object):
    def convertInteger(self, A, B):
        """
        :type A: int
        :type B: int
        :rtype: int
        """
        tmp = A ^ B
        count = 0
        for i in range(32):
            count += tmp & 1
            tmp = tmp >> 1
        return count

obj = Solution()
print(obj.convertInteger(826966453, -729934991))