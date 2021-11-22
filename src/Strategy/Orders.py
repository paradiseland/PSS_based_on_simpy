# coding=utf-8
# /usr/bin/Python3.6

"""
Author: Xingwei Chen
Email:cxw19@mails.tsinghua.edu.cn
date:2021/11/8 09:21
"""

from collections import OrderedDict

from Warehouse.Stocks import Stocks


class OrderDesignate:
    """
    Set the characteristics of order arrival.
    """

    def __init__(self, policy, stocks, fleet):
        self.policy: str = policy
        self.stocks: Stocks = stocks
        self.BP_fleet = fleet


class OrderPool:
    def __init__(self, release_policy):
        self.items = OrderedDict()
        self.release_policy = release_policy

    def append(self, order_entry):
        self.items[order_entry.name] = order_entry

    def pop(self, order_entry):
        self.items.pop(order_entry.name)

    def sort(self):
        """
        Sort the orders in the order pool
        :return:
        """
        sorted_orders = sorted(self.items, key=lambda x: x) # TODO:

