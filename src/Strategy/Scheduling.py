# coding=utf-8
# /usr/bin/Python3.6

"""
Author: Xingwei Chen
Email:cxw19@mails.tsinghua.edu.cn
date:2021/11/6 09:09
"""
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ORCSRS.PSS import PSS
    from Orders.OrderEntry import OrderEntry
from Robots.PSB import PSB


class SchedulingPolicy:
    """
    Description: currently, we has a available bp and order pool
                 we want to choose an strategy to
                    1. assign this psb to an order and
                    2. assign a place to the order.
    """

    def __init__(self, env):
        self.env: 'PSS' = env
        self.hit = {}  # 订单/sku命中料箱策略
        self.assign_bp = {}  # 空闲bp 绑定订单策略
        self.order_pool = env.order_pool

    def hit_stack(self, order: 'OrderEntry', strategy):
        """
        采取策略对指定订单进行命中
        :param order: 该订单/sku
        :param strategy: 采取策略
        :return: Stack info, including stack info and stack place. locked this stack immediately
        """
        pass

    def sort(self, **kwargs):
        """
        对订单池内订单进行优先级打分排序
        考虑因素包括: 已等待时间, 目标sku对应库存位置序列, 机器人位置信息,
        :return:
        """
        sorted(self.order_pool, key=lambda o: [o])
        return 0

    def select_target_for_order(self, order: 'OrderEntry', bp: 'PSB', order_sorted: bool):
        if order.type_ == 'S':
            # x_target = self.env.strategy['Storage'].store_line(order.sku_id)
            x, y, z = self.env.strategy['Storage'].store_to_line(order.sku_id, bp.place[0])
        elif not order_sorted:
            x_target = self.env.strategy['Storage'].random_retrieve_line(sku_id=order.sku_id)
            x, y, z, has_sku = self.env.strategy['Storage'].random_retrieve_from_line(x_target, order.sku_id)
        else:
            x, y, z, has_sku = self.env.strategy['Storage'].random_retrieve_from_line(bp.place[0], order.sku_id)
        return x, y, z

    def select_order_for_bp_0(self, bp):
        """
        【FIFO】根据到达时间排序，进行排序
        :return:
        """
        sorted_orders = sorted(self.order_pool.items.items(), key=lambda o: o[1].arrive_time)
        order_name, order_popped = sorted_orders[0]
        x, y, z = self.select_target_for_order(order_popped, bp, None)
        return order_name, order_popped, (x, y, z)

    def select_order_for_bp_1(self, bp: 'PSB'):
        """
        【FIFO considering tracks】考虑该车所在轨道的订单序列进行，找到该轨道内的顶部订单,且靠近工作台为优先，不考虑入库订单的track
        :return:
        """
        x, y = bp.place
        top0 = self.env.stocks.tier_x(0)
        sorted_orders = sorted(self.order_pool.items.items(), key=lambda o: o[1].arrive_time)
        sorted_orders_this_track = [(o[0], o[1]) for o in sorted_orders if
                                    o[0][0] == 'S' or o[1].sku_id in set(top0[x])]
        order_popped_name = sorted_orders_this_track[0][0] if sorted_orders_this_track else sorted_orders[0][0]
        order_popped = self.order_pool.items.pop(order_popped_name)
        x, y, z = self.select_target_for_order(order_popped, bp, sorted_orders_this_track)
        return order_popped_name, order_popped, (x, y, z)

    def select_order_for_bp_2(self, bp):
        """
        考虑次层任务
        :return:
        """
        x, y = bp.place
        top0 = self.env.stocks.tier_x(0)
        top1 = self.env.stocks.tier_x(1)
        self.select_target_for_order()

    def select_order_for_bp_3(self):
        pass

    def select_order_for_bp_4(self):
        pass
