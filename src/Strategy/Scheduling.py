# coding=utf-8
# /usr/bin/Python3.6

"""
Author: Xingwei Chen
Email:cxw19@mails.tsinghua.edu.cn
date:2021/11/6 09:09
"""
import copy
import random
from typing import TYPE_CHECKING, Tuple
import numpy as np

from ORCSRS.Config import NUM_OF_COLS
from Strategy.Orders import OrderPool
from util.run import timeit

if TYPE_CHECKING:
    from ORCSRS.PSS import PSS
from Orders.OrderEntry import OrderEntry, OutboundOrder, OutboundOrderEntry, InboundOrderEntry
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
        self.order_pool: OrderPool = env.order_pool

    def select_order(self, bp: 'PSB') -> Tuple['OrderEntry', tuple]:
        """
        target place hit rule: top first, travel_time(BP place with target place),  inline, dual_command
        :inline: considering the BP line
        order hit rule: 1: random; 2: FIFO, 3: Top then FIFO, 4.BP place
        :return: order, place
        """
        if self.env.action == 0:
            order_entry, target_place, batch_info = self.select_order_baseline_random(bp)
        elif self.env.action == 1:
            order_entry, target_place, batch_info = self.select_order_baseline_random(bp, inline=True)
        elif self.env.action == 2:
            order_entry, target_place, batch_info = self.select_order_baseline_random(bp, inline=True, travel_length=True)
        elif self.env.action == 3:
            order_entry, target_place, batch_info = self.select_order_baseline_random(bp, inline=True, travel_length=True,
                                                                          dual_command=True)
        elif self.env.action == 4:
            order_entry, target_place, batch_info = self.select_order_baseline_random(bp, inline=True, travel_length=True,
                                                                          dual_command=True, top_first=True)
        elif self.env.action == 5:
            order_entry, target_place, batch_info = self.select_order_baseline_random(bp, inline=True, travel_length=True,
                                                                          dual_command=True, top_first=True,
                                                                          consider_reshuffle=True)
        elif self.env.action == 6:
            order_entry, target_place, batch_info = self.select_order_baseline_fifo(bp)
        elif self.env.action == 7:
            order_entry, target_place, batch_info = self.select_order_baseline_fifo(bp, inline=True)
        elif self.env.action == 8:
            order_entry, target_place, batch_info = self.select_order_baseline_fifo(bp, inline=True, travel_length=True)
        elif self.env.action == 9:
            order_entry, target_place, batch_info = self.select_order_baseline_fifo(bp, inline=True, travel_length=True,
                                                                        dual_command=True)
        elif self.env.action == 10:
            order_entry, target_place, batch_info = self.select_order_baseline_fifo(bp, inline=True, travel_length=True,
                                                                        dual_command=True, top_first=True)
        elif self.env.action == 11:
            order_entry, target_place, batch_info = self.select_order_baseline_fifo(bp, inline=True, travel_length=True,
                                                                        dual_command=True, top_first=True,
                                                                        consider_reshuffle=True)
        else:
            raise ValueError(f"[Error], Wrong action number: {self.env.action}")
        return order_entry, target_place

    def select_target_for_order(self, order: 'OrderEntry', bp: 'PSB', tier_x_sku_set_inline, travel_length=False,
                                top_first=False):
        if order.type_ == InboundOrderEntry.type_:
            # x_target = self.env.strategy['Storage'].store_line(order.sku_id)
            x, y, z = self.env.strategy['Storage'].store_to_line(order.sku_id, bp.place[0])
        else:
            x, y, z, has_sku = self.select_target_considering_all(bp.place[0], order.sku_id, tier_x_sku_set_inline,
                                                                  travel_length, top_first)
        return x, y, z

    def select_target_considering_all(self, sku_id, x, tier_x_sku_set_inline, travel_length=False,
                                      top_first=False,
                                      inline=False):
        if inline and sku_id in tier_x_sku_set_inline[0].union(tier_x_sku_set_inline[1]):
            x, y, z, has_sku = self.env.strategy['Storage'].random_retrieve_from_line(x, sku_id)
        else:

            x_target = self.env.strategy['Storage'].random_retrieve_line(sku_id)
            x, y, z, has_sku = self.env.strategy['Storage'].random_retrieve_from_line(x_target, sku_id)
        return x, y, z, has_sku

    @timeit
    def select_order_baseline_random(self, bp: 'PSB' = None, inline=False, travel_length=False, dual_command=False,
                                     top_first=False, consider_reshuffle=False):
        """
        :param bp:
        :param inline:
        :param travel_length:
        :param dual_command:
        :param top_first:
        :param consider_reshuffle:
        :return: OrderEntry, designated place, batch or not
        """
        """
        thinking:
        Order popping: 
            - sort elements: 1. this command(dual command)[if need track-changing, it is not important], 2. top_first, 3.consider_reshuffle   
            - filter elements(-> sort, when available orders is not enough): 1. inline, 
            - randomize the order to pop up 
            ----------------------------------------------------------------------------------
            Top first has conflict/overlap with consider_reshuffle,
            order: inline > this command > top_first/consider_reshuffle
        Target place hit:  
            - sort elements: 1. travel length 2. top_first
            order: top_first/consider_reshuffle > travel_length
        """
        x, y = bp.place
        last_command = OutboundOrderEntry.type_ if y != -1 else InboundOrderEntry.type_
        tier_x_stocks, tier_x_stocks_inline, tier_x_sku_set_inline = {}, {}, {}
        if top_first:
            tier_x_stocks[0] = self.env.stocks.tier_x(0)
            tier_x_stocks_inline[0] = tier_x_stocks[0][x]  # collect the sku set of  tier0 containers.
            tier_x_sku_set_inline[0] = set(tier_x_stocks_inline[0])
        if consider_reshuffle:
            tier_x_stocks[1] = self.env.stocks.tier_x(1)
            tier_x_stocks_inline[1] = tier_x_stocks[1][x]
            tier_x_sku_set_inline[1] = set(tier_x_stocks_inline[1])
        sort_func = lambda order: (order[1].type_ == last_command if inline and dual_command else 0,
                                   order[1].sku_id not in tier_x_sku_set_inline[0] if top_first else 0,
                                   order[1].sku_id not in tier_x_sku_set_inline[0].union(
                                       tier_x_sku_set_inline[1]) if consider_reshuffle else 0)
        candidate_length = len(self.order_pool) // 4 if len(
            self.order_pool) // 4 >= 1 else 1  # reduce the candidate orders
        orders_sorted = sorted(self.order_pool.items.items(), key=sort_func)[:candidate_length]
        x_idx = (x, x + 1) if inline else (0, NUM_OF_COLS)
        if consider_reshuffle:  # combine two R orders together
            top0 = self.env.stocks.tier_x(0)
            top1 = self.env.stocks.tier_x(1)
        order_name = random.sample(orders_sorted, 1)[0][0]
        order_entry = self.order_pool.items.pop(order_name)
        target_place = self.select_target_for_order(order=order_entry, bp=bp,
                                                    tier_x_sku_set_inline=tier_x_sku_set_inline,
                                                    travel_length=travel_length)
        batch = True
        return order_entry, target_place, batch

    def select_order_baseline_fifo(self, bp: 'PSB' = None, inline=False, travel_length=False, dual_command=False,
                                   top_first=False,
                                   consider_reshuffle=False, ):
        """
        FIFO与DD(Due time) 调度规则是相同的
        """
        x, y = bp.place
        last_command = OutboundOrderEntry.type_ if y != -1 else InboundOrderEntry.type_
        tier_x_stocks, tier_x_stocks_inline, tier_x_sku_set_inline = {}, {}, {}
        if top_first:
            tier_x_stocks[0] = self.env.stocks.tier_x(0)
            tier_x_stocks_inline[0] = tier_x_stocks[0][x]  # collect the sku set of  tier0 containers.
            tier_x_sku_set_inline[0] = set(tier_x_stocks_inline[0])
        if consider_reshuffle:
            tier_x_stocks[1] = self.env.stocks.tier_x(1)
            tier_x_stocks_inline[1] = tier_x_stocks[1][x]
            tier_x_sku_set_inline[1] = set(tier_x_stocks_inline[1])
        sort_func = lambda order: (order[1].type_ == last_command if inline and dual_command else 0,
                                   order[1].sku_id not in tier_x_sku_set_inline[0] if top_first else 0,
                                   order[1].sku_id not in tier_x_sku_set_inline[0].union(
                                       tier_x_sku_set_inline[1]) if consider_reshuffle else 0)
        candidate_length = len(self.order_pool) // 4 if len(
            self.order_pool) // 4 >= 1 else 1  # reduce the candidate orders
        orders_sorted = sorted(self.order_pool.items.items(), key=sort_func)[:candidate_length]
        x_idx = (x, x + 1) if inline else (0, NUM_OF_COLS)
        if consider_reshuffle:  # combine two R orders together
            top0 = self.env.stocks.tier_x(0)
            top1 = self.env.stocks.tier_x(1)
        order_name = orders_sorted[0][0]
        order_entry = self.order_pool.items.pop(order_name)
        target_place = self.select_target_for_order(order=order_entry, bp=bp,
                                                    tier_x_sku_set_inline=tier_x_sku_set_inline,
                                                    travel_length=travel_length)
        batch = True
        return order_entry, target_place, batch

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

    def select_order_reshuffle_tier0(self, bp):
        """
        prefer to output top sku of certain stack
        """
        top_view = self.env.stocks.tier_x(0)

        pass

    def select_order_reshuffle_tier0_inline(self, bp):
        """
        prefer to output top sku of certain stack, in bp.line
        """
        top_view_x = self.env.stocks.tier_x(0)[bp.place[0]]

        pass

    def select_order_reshuffle_tier1(self, bp):
        """
        考虑次层任务
        :return:
        """
        x, y = bp.place
        top0_view = self.env.stocks.tier_x(0)
        top1_view = self.env.stocks.tier_x(1)
        self.select_target_for_order()
