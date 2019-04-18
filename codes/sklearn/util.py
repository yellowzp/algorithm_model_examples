#! /usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt


class Util(object):
    """
    utility tools
    """

    @classmethod
    def plot(cls, points, lines, title="test", x_label="X", y_label="Y", save_path=""):
        """
        绘制二维坐标图
        :param points:
        :param lines:
        :param title:
        :param x_label:
        :param y_label:
        :param save_path:
        :return:
        """
        point_color = "blue"
        color_list = ["green", "red", "cyan", "yellow", "purple", "springgreen", "orange", "lightcoral", "peru", "tan", "gold"]
        # points
        x_list, y_list = points
        plt.scatter(x_list, y_list, s=20, c=point_color, alpha=.5)
        # lines
        for idx, line in enumerate(lines):
            color = color_list[idx % len(color_list)]
            x_list, y_list = line
            plt.plot(x_list, y_list, color)
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.legend()
        plt.show()
        if save_path:
            plt.savefig(save_path)
        return True


