# coding=utf-8
# /usr/bin/Python3.6

"""
Author: Xingwei Chen
Email:cxw19@mails.tsinghua.edu.cn
date:2021/3/9 08:42
"""
from typing import Tuple, Any

from Stack import Stack

import numpy as np

# Here,
COL_OF_WAREHOUSE = 10
STACKS_OF_ONE_COL = 20
HEIGHT_OF_TIER = .33
NUM_OF_TIERS = 8
HEIGHT_AVAILABLE = 7
CAPACITY = COL_OF_WAREHOUSE * STACKS_OF_ONE_COL * NUM_OF_TIERS
AVAILABLE_CAPACITY = COL_OF_WAREHOUSE * STACKS_OF_ONE_COL * HEIGHT_AVAILABLE


class Warehouse:
    """
    Define a Warehouse class to get an available place and to save parameters of warehouse configuration.
    """

    def __init__(self):
        self.ratio = COL_OF_WAREHOUSE / STACKS_OF_ONE_COL
        self.stock_record = [[Stack(NUM_OF_TIERS) for _ in range(STACKS_OF_ONE_COL)] for _ in range(COL_OF_WAREHOUSE)]
        self.init_stocks()

    @property
    def current_storage(self) -> Tuple[np.ndarray, Any]:
        return np.sum(self.stacks_state), np.sum(self.stacks_state) / AVAILABLE_CAPACITY

    @property
    def stacks_state(self) -> np.ndarray:
        return np.array([self.stock_record[i][j].size_of_record_for_sync for i in range(COL_OF_WAREHOUSE) for j in range(STACKS_OF_ONE_COL)])

    def random_a_place_for_retrieval(self) -> Tuple[Tuple, int]:
        """
        ---------------
        Return x_y place, stack tier
        """
        stack_size = 0
        stock_share_of_each_stack: np.ndarray = self.stacks_state / sum(self.stacks_state)
        cumulative_sum = stock_share_of_each_stack.cumsum().reshape(COL_OF_WAREHOUSE, STACKS_OF_ONE_COL)
        place: Tuple[int, int] = (0, 0)
        stack_tier: int = 0

        while stack_size == 0:
            # random a stack to get good.
            left_of_chosen_stack = np.argwhere(cumulative_sum >= np.random.random())
            place = (left_of_chosen_stack[0][0], left_of_chosen_stack[0][1])
            stack_size = self.stock_record[place[0]][place[1]].size
            stack_tier = np.random.choice(self.stock_record[place[0]][place[1]].size_of_record_for_sync)

        if self.stock_record[place[0]][place[1]].size_of_record_for_sync > stack_tier:
            pass
        else:
            raise IndexError
        return place, stack_tier

    def init_stocks(self):
        tmp = np.ones(AVAILABLE_CAPACITY, dtype=np.bool)
        index_of_may_stocked = np.arange(AVAILABLE_CAPACITY)
        random_choose = np.random.choice(index_of_may_stocked, int(AVAILABLE_CAPACITY / 2), replace=False).tolist()
        for i in random_choose:
            tmp[i] = False
        stacks_height_from_random_choose = np.sum(tmp.reshape(-1, HEIGHT_AVAILABLE), axis=1).reshape(COL_OF_WAREHOUSE, STACKS_OF_ONE_COL)
        for line in range(COL_OF_WAREHOUSE):
            for stack_order in range(STACKS_OF_ONE_COL):
                self.stock_record[line][stack_order].init(stacks_height_from_random_choose[line, stack_order])