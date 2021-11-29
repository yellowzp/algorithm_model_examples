#! /usr/bin/env python
# -*- coding:utf-8 -*-
"""
@author: Vesper Huang
"""
class Solution(object):
    def setZeroes(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: None Do not return anything, modify matrix in-place instead.
        """
        row_num = 0
        col_num = 0
        if matrix:
            row_num = len(matrix)
            if matrix[0]:
                col_num = len(matrix[0])
        row_to_zero = []
        col_to_zero = []
        for i in range(row_num):
            for j in range(col_num):
                if matrix[i][j] == 0:
                    row_to_zero.append(i)
                    col_to_zero.append(j)
        row_to_zero = list(set(row_to_zero))
        col_to_zero = list(set(col_to_zero))
        # print(row_to_zero, col_to_zero)
        for i in range(row_num):
            if i in row_to_zero:
                matrix[i] = [0] * col_num
            else:
                for j in range(col_num):
                    if j in col_to_zero:
                        matrix[i][j] = 0

obj = Solution()
print(obj.setZeroes([[0,1,2,0],[3,4,5,2],[1,3,1,5]]))