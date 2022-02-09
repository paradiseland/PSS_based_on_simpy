# coding=utf-8
# /usr/bin/Python3.6

"""
Author: Xingwei Chen
Email:cxw19@mails.tsinghua.edu.cn
date:2021/11/8 09:21
"""

from collections import OrderedDict
from typing import TYPE_CHECKING
from ORCSRS.Config import order_pool_size

if TYPE_CHECKING:
    from Orders.OrderEntry import OrderEntry
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
        if isinstance(order_entry, str):
            return self.items.pop(order_entry)
        elif isinstance(order_entry, OrderEntry):
            return self.items.pop(order_entry.name)
        else:
            raise ValueError

    def foreXorders(self, x, t):
        """
        Get the fore [order pool size] orders.
        :param x:
        :param t:
        :return:
        """
        sorted_order = sorted(self.items.values(), key=lambda o: -o.has_waiting_since(t))
        return sorted_order[:x]

    def __len__(self):
        return len(self.items)

    def __str__(self):
        return f"{len(self.items)} orders"

    @property
    def sku_dict(self):
        return {k: v.sku_id for k, v in self.items.items()}
