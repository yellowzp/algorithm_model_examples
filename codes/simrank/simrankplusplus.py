#! /usr/bin/env python
# -*- coding:utf-8 -*-
"""
author: Vesper Huang
"""
import numpy as np
import math


class SimRankPlusPlus(object):
    """
    朴素SimRank实现类

    Attributes:
        C: 阻尼系数
        vertex_list: 节点集合
        v_num: 节点数
        w_matrix: 转移概率矩阵
        s_matrix: 相似度矩阵
        e_matrix: 证据矩阵
    """

    def __init__(self, C, bi_graph_file):
        """

        :param C: 阻尼系数
        :param bi_graph_file: 二部图文件路径
        """
        self.C = C
        self.vertex_list = []
        self.v_a_list = []
        self.v_b_list = []
        self.v_num = 0
        self.w_matrix = self._read_bi_graph(bi_graph_file)
        self.s_matrix = np.identity(self.v_num)
        self.e_matrix = self._init_evidence()

    def _read_bi_graph(self, bi_graph_file):
        """

        :param bi_graph_file:
        :return w_matrix:
        """
        vertex_dict = {}
        edge_list = []

        f = open(bi_graph_file, "r")
        for line in f:
            line_seg = line.strip().split("\t")
            v1 = line_seg[0]
            v2 = line_seg[1]
            click = line_seg[2]
            if v1 not in vertex_dict:
                vertex_dict[v1] = self.v_num
                self.vertex_list.append(v1)
                self.v_a_list.append(self.v_num)
                self.v_num += 1
            if v2 not in vertex_dict:
                vertex_dict[v2] = self.v_num
                self.vertex_list.append(v2)
                self.v_b_list.append(self.v_num)
                self.v_num += 1
            edge_list.append((vertex_dict[v1], vertex_dict[v2], click))
        f.close()

        # 初始化为二部图连接矩阵
        w_matrix = np.zeros([self.v_num, self.v_num], dtype=float)
        for edge in edge_list:
            w_matrix[edge[0], edge[1]] += edge[2]
            w_matrix[edge[1], edge[0]] += edge[2]

        # 每一行归一化，变换为转移概率
        for idx in xrange(self.v_num - 1):
            total = np.sum(w_matrix[idx])
            w_matrix[idx] = w_matrix[idx] / total

        # 每一列乘上方差项
        for idx in xrange(self.v_num - 1):
            var = np.var(w_matrix[:, idx])
            tmp = math.exp(-var)
            w_matrix[:, idx] = tmp * w_matrix[:, idx]

        # 修正对角线上的值
        for idx in xrange(self.v_num - 1):
            w_matrix[idx, idx] = 0
            tmp = np.sum(w_matrix[idx])
            w_matrix[idx, idx] = 1 - tmp

        # for key, value in vertex_dict.items():
        #     print key, value
        # print w_matrix
        return w_matrix

    def _init_evidence(self):
        pass

    def run(self, iter_num):
        # print self.s_matrix
        for idx in xrange(iter_num):
            tmp = self.C * np.dot(np.dot(self.w_matrix.transpose(), self.s_matrix), self.w_matrix)
            new_s = tmp + np.identity(self.v_num) - np.diag(np.diag(tmp))
            self.s_matrix = new_s
            # print self.s_matrix

    def print_sim(self):
        print "set a:"
        for idx1, item1 in enumerate(self.v_a_list):
            for item2 in self.v_a_list[idx1 + 1:]:
                print "sim(%s, %s)=%.4f" % (self.vertex_list[item1], self.vertex_list[item2],
                                            self.s_matrix[item1, item2])

        print "set b:"
        for idx1, item1 in enumerate(self.v_b_list):
            for item2 in self.v_b_list[idx1 + 1:]:
                print "sim(%s, %s)=%.4f" % (self.vertex_list[item1], self.vertex_list[item2],
                                            self.s_matrix[item1, item2])


if __name__ == "__main__":
    C = 0.8
    bi_graph_file = "../../data/simrank_test.txt"
    iter_num = 5
    obj = SimRankPlusPlus(C, bi_graph_file)
    obj.run(iter_num)
    # obj.print_sim()

