# coding=utf-8
# /usr/bin/Python3.6

"""
Author: Xingwei Chen
Email:cxw19@mails.tsinghua.edu.cn
date:2021/6/7 14:27
"""
import logging
from typing import Tuple

import numpy as np
from Config import *

np.random.seed(42)
class StackError(BaseException):
    pass


class Stocks:
    """
    用来存储仓库内的库存信息，采用numpy.ndarray实现
    """

    def __init__(self, shape: Tuple[int, int, int], rate):
        """基于numpy.ndarray来参数化生成仓库库存
        ---------------------------------------------------
        其中数字代表货物编码, 当前仅存在1一种货物, 0表示该位置为空
        :param shape: 仓库形状
        width: 几列轨道, length: 每列轨道的堆塔数量, height: 每个堆塔的可堆放层数
        :param rate: 仓库现有货物比率
        """
        self.shape = shape
        self.total_place = np.prod(shape)
        self.s: np.ndarray = np.zeros(shape, dtype=np.uint8)
        # 为每个堆塔提供一个锁, 一般锁位于每个堆塔顶层货物的下一层, 但当顶层被确定出入货时, 锁在顶层+1层
        self.sync_ = np.ones(shape[:2], dtype=np.bool)
        self.initialize(rate)
        logging.info(f"Stocks initializing...\nshape:{self.shape}, occupied rate: {self.s_rate:.2%}, avail stacks:{np.sum(self.sync_)}\nStocks details (2d gridview):\n{self.s_2d}")

    def initialize(self, rate):
        """
        随机初始化库存信息
        :param rate: 库存占用率
        """
        rand_int = np.random.normal(loc=rate * self.shape[2], scale=2, size=self.shape[:2]).astype(np.uint8)
        rand_int[rand_int > self.shape[2]] = self.shape[2]
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                self.s[i, j][:rand_int[i, j]] = 1

    @property
    def s_2d(self):
        return self.s.sum(axis=2, dtype=np.uint8)



    @property
    def s_1d(self) -> np.ndarray:
        return self.s.sum(axis=2, dtype=np.uint16).sum(axis=1)



    @property
    def s_rate(self):
        return self.s.sum() / self.total_place

    @property
    def s_state(self):
        """
        呈现库存锁定状况
        :return: 返回所有的锁定位置以及锁定堆塔数量
        """
        locked = np.argwhere(self.s_2d != self.s_2d_sync)
        return locked, locked.shape[0]

    def rand_S_place(self, psb_idle=None) -> Tuple[int, int, int]:
        """
        随机一个
        :param psb_idle: psb车队的空闲情况
        :return: 返回 (x, y)
        """
        # avail_xys = np.argwhere((self.s_2d < Warehouse.num_of_tiers - 1) & (self.s_2d == self.s_2d_sync))
        avail_xys = np.argwhere((self.s_2d < self.shape[2] - 1) & self.sync_)
        avail_xys_list = avail_xys.tolist()
        avail_xys_list.sort(key=lambda xy: self.s_2d[xy[0], xy[1]], reverse=False)
        assert len(avail_xys_list) > 1, "入库速率已经大于出库速率，没有同步可用堆塔用来出库"
        x, y = avail_xys_list[np.random.randint(len(avail_xys_list) // 2)]
        z = self.s_2d[x, y]
        self.sync_[x, y] = False
        # self.sync[x, y] += 1
        # print(f"Random-S:{(x, y, z)}, stack: {self.s[x, y]}")
        # print(f"                     locked: {(x, y)}, {self.sync_[x,y]}")
        return x, y, z

    def rand_R_place(self, sku_id=None) -> Tuple[int, int, int]:
        """
        随机一个出库位置, 无需考虑车队情况, 仅考虑仓库是否有货情况, 在随机位置时，将完成该库位的同步操作
        :return: 返回 (x, y)
        """
        # 对于已被锁定的货物，采取放弃随机的操作
        # avail_xys = np.argwhere((self.s_2d > 0) & (self.s_2d == self.s_2d_sync))
        if sku_id is None:
            avail_xys = np.argwhere((self.s_2d > 0) & self.sync_)
        else:
            avail_xys = np.argwhere((self.s_2d > 0) & self.sync_)
        x, y = avail_xys[np.random.randint(avail_xys.shape[0])]
        z = np.random.randint(self.s_2d[x, y])
        self.sync_[x, y] = False
        # self.sync[x, y] -= 1
        # print(f"Random-R:{(x, y, z)}, stack: {self.s[x, y]}")
        # print(f"                     locked: {(x, y)}, {self.sync_[x, y]}")

        return x, y, z

    def rand_R_x(self, sku_id=None):
        if sku_id is None:
            avail_xys = np.argwhere((self.s_2d > 0) & self.sync_)
        else:
            avail_xys = np.argwhere((self.s_2d > 0) & self.sync_)
        logging.debug(f'可取货堆塔数{len(avail_xys)}, 锁定堆塔数:{np.sum(~self.sync_)}, 空堆塔数:{np.sum(self.s_2d == 0)}')  # 排除空堆塔与被锁定堆塔
        x, y = avail_xys[np.random.randint(avail_xys.shape[0])]
        return x

    def rand_R_from_x(self, x, sku_id=None):
        if sku_id is None:
            avail_xys = np.argwhere((self.s_2d[x, :] > 0) & self.sync_[x, :])
        else:
            avail_xys = np.argwhere((self.s_2d[x, :] > 0) & self.sync_[x, :])
        y = avail_xys[np.random.randint(avail_xys.shape[0])][0]
        z = np.random.randint(self.s_2d[x, y])
        return x, y, z

    def S(self, x, y, z=None, is_S=True):
        """
        判断合理性, 入库修改库存
        """
        if z is None:
            z = self.s[x, y].sum()
        # try:
        #     assert z == 0 or (self.s[x, y][:z] == 1).all(), "堆塔入库位置下层非全部有货"
        #     assert (self.s[x, y][z:] == 0).all(), "堆塔入库位置上层非空"
        #     assert 0 <= x < Warehouse.num_of_cols
        #     assert 0 <= y < Warehouse.stacks_of_one_col
        #     assert 0 <= z < Warehouse.num_of_tiers - 1
        # except AssertionError:
        #     raise StackError(f"堆塔发生错误\n此时堆塔库存信息: {self.s[x, y]}\n存货位置: ({x}, {y}, {z})")
        # else:
        #     self.s[x, y, z] = 1
        #     self.sync[x, y] += 1
        self.s[x, y, z] = 1
        if is_S:
            self.sync_[x, y] = True
        # print(f"Finish-S:{(x, y, z)}, stack: {self.s[x, y]}")
        # print(f"                     locked: {(x, y)}, {self.sync_[x, y]}")

    def R(self, x, y, z=None, is_R=True):
        """
        依据取货位置来进行取货, 判断合理性, 修改库存
        :param z:
        :param y:
        :param x:
        :param is_R:
        """
        if z is None:
            z = int(self.s[x, y].sum() - 1)
        # try:
        #     assert (self.s[x, y][:z + 1] == 1).all()
        #     assert (self.s[x, y][z + 1:] == 0).all()
        # except AssertionError:
        #     print(f"堆塔发生错误\n此时堆塔库存信息: {self.s[x, y]}\n取货位置: ({x}, {y}, {z})")
        #     raise StackError
        # else:
        #     self.s[x, y, z] = 0
        self.s[x, y, z] = 0
        if is_R:
            self.sync_[x, y] = True
            # self.sync[x, y] -= 1
        # print(f"Finish-R:{(x, y, z)}, stack: {self.s[x, y]}")
        # print(f"                     locked: {(x, y)}, {self.sync_[x, y]}")


if __name__ == '__main__':
    test = Stocks(shape=(10, 20, 7), rate=0.5)
    s = test.rand_S_place()
    r = test.rand_R_place()
    rr = test.rand_R_x()
    print(f"单列库存:\n{test.s_1d}")
    print(f"平面库存:\n{test.s_2d}")
    print(f"随机入库位置: {s}, 库存: {test.s[s]}")
    print(f"随机出库位置: {r}, 库存:{test.s[r[:2]]}")
