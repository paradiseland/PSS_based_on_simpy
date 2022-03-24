# coding=utf-8
# /usr/bin/Python3.6

"""
Author: Xingwei Chen
Email:cxw19@mails.tsinghua.edu.cn
date:2021/11/6 09:09
"""
import logging
import random
from typing import TYPE_CHECKING, Union

import numpy as np
import pandas as pd

from ORCSRS.Config import WAREHOUSE_INIT_RATE, NUM_OF_TIERS, NUM_OF_COLS, STACKS_OF_ONE_COL, NUM_OF_SKU, zoned_coe, \
    zoned_first_class_cumsum_percent, AVAIL_PLACE, RANDOM, ZONED, TOP_FIRST, DETERMINED, stock_threshold
from ORCSRS.Exception import *
from Orders.ordersGDS import bucket

if TYPE_CHECKING:
    from Warehouse.Stocks import Stocks


class StoragePolicy:

    def __init__(self, policy, stocks):
        self.policy: str = policy
        self.order_hit_policy = RANDOM
        self.stocks: Stocks = stocks
        self.first_class_sku_info = {}
        self.other_class_sku_info = {}
        self.zoned_info = {}
        self.determined_layout = None
        self.zoned_y = []
        self.sku_part = None
        self.df_bin = bucket(NUM_OF_SKU) if NUM_OF_SKU > 1 else None
        self.initialize()

    def initialize(self):
        """
        根据不同存储策略和问题类型，随机初始化库存信息
        """
        if NUM_OF_SKU == 1 and self.policy == RANDOM:
            self.init_single_sku_random_store()
            self.sku_part = {1: 1}
        elif NUM_OF_SKU > 1 and self.policy == RANDOM:
            self.sku_part = self.init_multi_sku_random_store()
        elif NUM_OF_SKU > 1 and self.policy == ZONED:
            part, self.zoned_info = self.init_multi_sku_zoned_store()
            self.sku_part = part
        elif NUM_OF_SKU > 1 and self.policy == DETERMINED:
            part, self.determined_layout = self.init_multi_sku_determined_store()
            self.sku_part = part
        else:
            raise ValueError(f"Error Storage Info, ref:NUM_OF_SKU:{NUM_OF_SKU}, policy:{self.policy}")

    def init_single_sku_random_store(self):
        """
        Single-sku and random storage policy.
        randomize generate stocks ~ Norm(μ=0.5 * 8, σ=2)
        """
        rand_int = np.random.normal(loc=WAREHOUSE_INIT_RATE * NUM_OF_TIERS, scale=2,
                                    size=(NUM_OF_COLS, STACKS_OF_ONE_COL)).astype(np.uint8)
        rand_int[rand_int > NUM_OF_TIERS] = NUM_OF_TIERS
        for i in range(NUM_OF_COLS):
            for j in range(STACKS_OF_ONE_COL):
                self.stocks.s[i, j][:rand_int[i, j]] = 1
        self.other_class_sku_info = {1: []}

    def init_multi_sku_random_store(self):
        pss_sku = self.multi_sku_info()
        prob = pss_sku / pss_sku.sum()
        # modify_index = 7
        # prob2 = prob * 2
        # prob3 = prob2[:modify_index] / prob2[:modify_index].sum() * (1 - prob2[modify_index:].sum())
        # modified_prob = sorted(np.hstack([prob3, prob2[modify_index:]]), reverse=True)
        rand_int = np.random.normal(loc=WAREHOUSE_INIT_RATE * NUM_OF_TIERS, scale=2,
                                    size=(NUM_OF_COLS, STACKS_OF_ONE_COL)).astype(np.uint32)
        rand_int[rand_int > NUM_OF_TIERS] = NUM_OF_TIERS
        for i in range(NUM_OF_COLS):
            for j in range(STACKS_OF_ONE_COL):
                for k in range(rand_int[i, j]):
                    self.stocks.s[i, j, k] = np.random.choice(range(1, NUM_OF_SKU + 1), p=prob)
        self.other_class_sku_info = {i: [] for i in range(NUM_OF_SKU)}
        return prob

    def init_multi_sku_zoned_store(self) -> dict:
        """
        In the method, we classify 2-8 good to different section, assign them to different storage target_xyz.
        :return: zone_info zoned storage information record, like {1: [1,2,3], 2:[4,5,6], 3:[7, 8], 4: 0[]}
        if there are any elements ,meaning that this sku_id is stored in this stack order in one column
        """
        pss_sku = self.multi_sku_info()
        part = pss_sku / pss_sku.sum()
        first_class_sku = np.arange(1, np.argwhere(part.cumsum() > zoned_first_class_cumsum_percent)[0, 0] + 1)
        other_class_sku = np.setdiff1d(np.arange(1, NUM_OF_SKU + 1), first_class_sku)
        zoned_place = np.asarray(
            [0] + np.ceil(part[:first_class_sku[-1] + 1] * STACKS_OF_ONE_COL * WAREHOUSE_INIT_RATE * zoned_coe).astype(
                np.int).tolist()).cumsum()
        z = [list(range(zoned_place[i - 1], zoned_place[i])) for i in range(1, zoned_place.shape[0])]
        self.first_class_sku_info = dict(zip(first_class_sku, z))
        self.other_class_sku_info = dict(zip(other_class_sku, other_class_sku.shape[0] * [[]]))
        zone_info = {**self.first_class_sku_info, **self.first_class_sku_info}
        from functools import reduce
        from operator import add
        self.zoned_y = reduce(add, list(zone_info.values()))
        rand_int = np.random.normal(loc=WAREHOUSE_INIT_RATE * NUM_OF_TIERS, scale=2,
                                    size=(NUM_OF_COLS, STACKS_OF_ONE_COL - self.zoned_y[-1] - 1)).astype(np.uint8)
        rand_int[rand_int > NUM_OF_TIERS] = NUM_OF_TIERS
        for i in range(NUM_OF_COLS):
            for j in range(self.zoned_y[-1] + 1, STACKS_OF_ONE_COL):
                for k in range(rand_int[i, j - self.zoned_y[-1] - 1]):
                    self.stocks.s[i, j, k] = np.random.choice(range(first_class_sku[-1] + 1, NUM_OF_SKU + 1),
                                                              p=part[first_class_sku.size:] / part[
                                                                                              first_class_sku.size:].sum())
        for sku_id, store_place in self.first_class_sku_info.items():
            qty = round(AVAIL_PLACE * WAREHOUSE_INIT_RATE * part[sku_id - 1] / NUM_OF_COLS)
            y_1, y_2 = divmod(qty, NUM_OF_TIERS)
            self.stocks.s[:, store_place[0]: store_place[0] + y_1, :] = sku_id
            self.stocks.s[:, store_place[0] + y_1, :y_2] = sku_id
        return part, zone_info

    def init_multi_sku_determined_store(self) -> dict:
        """
        In the method, we classify 2-8 good to different section, assign them to different storage target_xyz.
        :return: zone_info zoned storage information record, like {1: [1,2,3], 2:[4,5,6], 3:[7, 8], 4: 0[]}
        if there are any elements ,meaning that this sku_id is stored in this stack order in one column
        """
        pss_sku = self.multi_sku_info()
        part = pss_sku / pss_sku.sum()
        stack_qty = np.round(part * NUM_OF_COLS * STACKS_OF_ONE_COL).astype(int)
        self.stack_qty = stack_qty
        stack_qty[0] -= stack_qty.sum() - NUM_OF_COLS * STACKS_OF_ONE_COL
        stack_layout = np.hstack([np.asarray([sku_id + 1]).repeat(qty) for sku_id, qty in enumerate(stack_qty)])
        np.random.shuffle(stack_layout)
        stack_layout = stack_layout.reshape((NUM_OF_COLS, STACKS_OF_ONE_COL))

        for x in range(NUM_OF_COLS):
            for y in range(STACKS_OF_ONE_COL):
                self.stocks.s[x, y, :int(WAREHOUSE_INIT_RATE * NUM_OF_TIERS)] = stack_layout[x, y]
        part = (pd.Series(stack_layout.ravel()).value_counts() / stack_layout.size).values
        return part, stack_layout

    def store_line(self, sku_id, limit=None):
        """
        Before getting a BP robot, one line will be designated to this inbound task.
        :return: line index / x
        """
        if limit is None:
            limit = []
        if self.policy == ZONED and sku_id in self.first_class_sku_info.keys():
            # multi-sku, zoned policy | top sku
            zoned_y = self.first_class_sku_info[sku_id]
            avail_place = len(zoned_y) * NUM_OF_TIERS - np.count_nonzero(self.stocks.s[:, zoned_y], axis=2).sum(axis=1)
            prob = avail_place / avail_place.sum()
            return np.random.choice(list(range(NUM_OF_COLS)), p=prob)
        elif self.policy == DETERMINED:
            # multi-sku, determined
            try:
                avail_x = ((self.determined_layout == sku_id) & (
                        self.stocks.s_2d < NUM_OF_TIERS) & self.stocks.sync_).sum(axis=1)
                prob = avail_x / avail_x.sum()
                return np.random.choice(list(range(NUM_OF_COLS)), p=prob)
            except ValueError:
                avail_x = (self.determined_layout == sku_id)
                return 1
        else:
            # random policy
            return np.random.choice(NUM_OF_COLS)

    def store_to_line(self, sku_id, x):
        """
        Store specific sku_id to line x
        :return:
        """
        if self.policy == RANDOM:  # random storage policy
            x, y, z = self.random_store_to(sku_id, x)
        elif self.policy == ZONED:  # zoned storage policy
            x, y, z = self.zoned_store_to(sku_id, x)
        elif self.policy == DETERMINED:  # determined storage policy
            x, y, z = self.determined_store_to(sku_id, x)
        else:  # non-designated policy
            x, y, z = None, None, None
        return x, y, z

    def zoned_store_to(self, sku_id, x):
        """
        non-first class sku_id will be randomly stored, first class will be determined stored.
        """
        if sku_id in self.first_class_sku_info.keys():
            x, y, z = self.zoned_determined_store_to(sku_id, x)
        elif sku_id in self.other_class_sku_info.keys():
            x, y, z = self.random_store_to(sku_id, x)
        else:
            logging.error(f"Wrong sku_id{sku_id}")
            x, y, z = None, None, None
        self.stocks.sync_[x, y] = False
        return x, y, z

    def zoned_determined_store_to(self, sku_id, x):
        assert sku_id in self.first_class_sku_info.keys()
        avail_y = [index for index, line_qty in zip(self.first_class_sku_info[1],
                                                    np.count_nonzero(self.stocks.s[x, self.first_class_sku_info[1], :],
                                                                     axis=1)) if line_qty < NUM_OF_TIERS]
        if len(avail_y) > 0:
            try:
                y = avail_y[np.random.choice(len(avail_y))]
                z = self.stocks.s_2d[x, y]
                return x, y, z
            except ValueError:
                logging.info(f"Wrong determined store place:{x}, sku_id:{sku_id}")
                return self.zoned_store_to(sku_id, x=self.store_line(sku_id))
        else:
            # FIXME
            y = random.randrange(STACKS_OF_ONE_COL)
            z = self.stocks.s_2d[x, y]
            if z == NUM_OF_TIERS:
                z -= 1
            return x, y, z
    def random_store_to(self, sku_id, x):
        """
        Generate a storage place randomly.
        :param sku_id: designated sku id
        :param x: designated line x
        :return: storage place, (line[x], stack[y], tier[z])
        """
        avail_y = np.setdiff1d(list(range(STACKS_OF_ONE_COL)), self.zoned_y)
        avail_y = np.argwhere((self.stocks.s_2d[x, avail_y] < NUM_OF_TIERS) & self.stocks.sync_[x, avail_y])
        avail_y_list = avail_y.tolist()
        assert len(avail_y_list) >= 1, f"[Error] StockError, No available storage place in line [{x}]."

        y = avail_y_list[np.random.randint(len(avail_y_list))][0]
        if self.policy != RANDOM:
            y += self.zoned_y[-1] + 1  # TODO:
        z = self.stocks.s_2d[x, y]
        self.stocks.sync_[x, y] = False  # lock this stack
        return x, y, z

    def determined_store_to(self, sku_id, x, x_not_available=None):  # TODO:  BUG 这儿的问题，确定性存储时，库位已满，需要重置订单
        """

        :param x_not_available:
        :param sku_id:
        :param x:
        :return:
        """
        if x_not_available is None:
            x_not_available = set()
        status = 'normal'  # normal, repeat, full
        avail_y = np.argwhere((self.determined_layout[x] == sku_id) & (self.stocks.s_2d[x] < NUM_OF_TIERS) & (
            self.stocks.sync_[x])).ravel()
        if avail_y.size == 0:
            pass
        try:
            y = np.random.choice(avail_y)
        except ValueError:
            x_not_available.add(x)
            logging.info(f"[Error] determined place full, sku_id-{sku_id}")
            dic = dict(zip(self.stocks.value_counts[::-1][0], self.stocks.value_counts[::-1][1]))
            if sku_id in dic:
                status = 'repeat'
            else:
                status = 'full'
        if status == 'normal':
            z = self.stocks.s_2d[x, y]
            assert self.determined_layout[x, y] == sku_id
            return x, y, z
        elif status == 'repeat' and len(x_not_available) < NUM_OF_COLS:
            another_line = np.random.choice(list(set(range(NUM_OF_COLS)) - x_not_available))
            return self.determined_store_to(sku_id, another_line, x_not_available)
        else:

            raise DedicatedStorageAreaFullException(f"sku-{sku_id} determined store area full")

    def random_retrieve(self, sku_id):
        """
        Generate the place of retrieval considering sku id randomly.
        :param sku_id: outbound sku id
        :return:
        """
        status = 1  # return code. 1: get normal place, 0: abnormal retrieve place.
        avail_xyz = np.argwhere(
            (self.stocks.s == sku_id) & np.expand_dims(self.stocks.sync_, axis=2).repeat(NUM_OF_TIERS, axis=2))
        try:
            x, y, z = avail_xyz[np.random.choice(avail_xyz.shape[0])]
        except ValueError:
            status = 0
            logging.error(f"[Error] StockEmptyException: Quantity of {sku_id} is 0.")
            logging.debug(
                f"[Error] StockEmptyException: sku_id-{sku_id}: {np.count_nonzero(self.stocks.s == sku_id)} left in the system.")
            return 0, 0, 0, status
        else:
            return x, y, z, status

    def random_retrieve_line(self, sku_id):
        """
        randomize the retrieve line first.
        :param sku_id: outbound sku id
        :return: line_no
        """
        avail_x = np.argwhere(
            (self.stocks.s == sku_id) & np.expand_dims(self.stocks.sync_, axis=2).repeat(NUM_OF_TIERS, axis=2))[:, 0]
        try:
            x = np.random.choice(avail_x)
        except ValueError:
            logging.error(f"[Error] StockEmptyException: sku_id: {sku_id}")
            return -1  # not avail sku {sku_id}
        return x

    def random_retrieve_from_line(self, x, sku_id):
        """
        In the line x, get a random retrieval place whose sku id is sku_id.
        :param x: line_no
        :param sku_id
        :return: x,y,z place and status
        """
        avail_yz = np.argwhere(self.stocks.s[x] == sku_id)  # all the avail place
        if self.policy == DETERMINED:
            try:
                y = np.random.choice(avail_yz[:, 0].ravel())
                z = self.stocks.s_2d[x, y] - 1
                return x, y, z, 1
            except ValueError:
                return self.random_retrieve_from_line(x=self.store_line(sku_id), sku_id=sku_id)
        stack_has_sku, stack_has_sku_on_top = self.line_x_sku_info(x, sku_id)
        if not stack_has_sku.any():
            return self.random_retrieve(sku_id)
        else:  # will modify the designated x.
            if self.order_hit_policy == RANDOM:
                y = np.random.choice(stack_has_sku)
            elif self.order_hit_policy == TOP_FIRST:
                y = np.random.choice(stack_has_sku_on_top) if stack_has_sku_on_top else np.random.choice(stack_has_sku)
            else:
                raise StrategyConfException(f"[Error] StrategyConfException: self.order_hit_policy is not in set.")
        tier = np.argwhere(self.stocks.s[x, y] == sku_id).ravel()[-1]  # get the upper retrieval tier.
        self.stocks.sync_[x, y] = False
        return x, y, tier, 1

    def line_x_sku_info(self, x, sku_id):
        """
        sku quantity and state info in the line x.
        :param x: line_no
        :param sku_id: sku id queried
        :return: Stack index which has sku or has sku on top.
        """
        res = np.empty((STACKS_OF_ONE_COL, 3))  # matrix for relation between target sku with this col.
        # res: [index, isHasSku, isHasSkuOnTop]
        for idx, stack in enumerate(self.stocks.s[x]):
            has_sku = int(sku_id in stack)
            has_sku_on_top = int(stack[::-1][(stack[::-1] != 0).argmax(axis=0)] == sku_id)
            res[idx, :] = [idx, has_sku,
                           has_sku_on_top]  # (index, contain target sku or not, target sku is on top), in this column.
        stack_has_sku_on_top = np.argwhere(res[:, 2] == 1).ravel()
        stack_has_sku = np.argwhere(res[:, 1] == 1).ravel()
        return stack_has_sku, stack_has_sku_on_top

    def multi_sku_info(self):
        """
        Compute stocks value counts.
        :return:
        """
        pss_sku = self.df_bin.value_counts().values
        return pss_sku

    def order_arrive(self, inbound=False):
        """
        Outbound orders are randomly generated by pareto distribution.
        while inbound orders are generated considering current stocks information to avoid certain used up sku.
        :param inbound: whether a inbound task or not
        :return: sku id of this task.
        """
        # inbound = False
        # if inbound:
        #     stock_info = self.stocks.value_counts
        #     danger_sku = np.argwhere(stock_info < stock_threshold) + 1
        #     if danger_sku.shape[0] > 0:  # random the dangerous sku
        #         return np.random.choice(danger_sku.ravel())
        #     elif self.policy == DETERMINED:  #
        #         non_overflow_sku = (np.argwhere(((self.stack_qty * NUM_OF_TIERS) - stock_info) > 10) + 1).ravel()
        #         return np.random.choice(non_overflow_sku.ravel(), p=self.sku_part[non_overflow_sku - 1] / (
        #             self.sku_part[non_overflow_sku - 1]).sum())
        #     else:
        #         return np.random.choice(a=np.arange(1, len(self.sku_part) + 1), p=self.sku_part)
        # else:
        #     return np.random.choice(a=np.arange(1, len(self.sku_part) + 1), p=self.sku_part)
        return np.random.choice(a=np.arange(1, len(self.sku_part) + 1), p=self.sku_part)

    def explore_another_lines(self, x_dir=True, rule_out=None):
        if rule_out is None:
            rule_out = set()
        return list(set(range(NUM_OF_COLS)) - rule_out) if x_dir else list(set(range(STACKS_OF_ONE_COL)) - rule_out)
