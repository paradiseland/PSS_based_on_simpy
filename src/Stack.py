# coding=utf-8
# /usr/bin/Python3.6

"""
Author: Xingwei Chen
Email:cxw19@mails.tsinghua.edu.cn
date:2021/3/9 08:42
"""
import random

from typing import List
import math

LENGTH = 0.6
HEIGHT_OF_ONE_TIER = .33


class Stack:
    """
    we set a twin data saver for the lag of picking.
    """
    max_stacks_num = 7
    tolerate_coefficient = .7
    height_available = math.ceil(max_stacks_num * tolerate_coefficient)

    def __init__(self, height):
        self.stack: List[int] = []
        self.record_for_sync: List[int] = []
        self.lock: bool = False

    def init(self, certain_number_of_initialization: int):
        self.stack = [1] * certain_number_of_initialization
        self.record_for_sync = [1] * certain_number_of_initialization

    @property
    def size(self) -> int:
        return len(self.stack)

    @property
    def size_of_record_for_sync(self) -> int:
        return len(self.record_for_sync)

    def is_empty(self) -> bool:
        return len(self.stack) == 0

    def is_peek(self, tier) -> bool:
        return tier == self.size - 1

    def pop(self):
        return self.stack.pop()

    def pop_from_record_for_sync(self):
        return self.record_for_sync.pop()

    def push(self, item: int):
        self.stack.append(item)

    def push_into_record_for_sync(self, item: int):
        self.record_for_sync.append(item)

    def sync(self):
        self.record_for_sync = self.stack
