# coding=utf-8
# /usr/bin/Python3.6

"""
Author: Xingwei Chen
Email:cxw19@mails.tsinghua.edu.cn
date:2021/6/7 14:27
"""
from typing import Tuple

import pandas as pd

from ORCSRS.Config import *
from util.run import timeit
from Strategy.Storage import StoragePolicy

np.random.seed(42)


class StackError(BaseException):
    pass


class Stocks:
    """
    用来存储仓库内的库存信息，采用numpy.ndarray实现
    """

    def __init__(self, shape: Tuple[int, int, int], env):
        """基于numpy.ndarray来参数化生成仓库库存
        ---------------------------------------------------
        其中数字代表货物编码, 0表示该位置为空
        :param shape: 仓库形状. width: 几列轨道, length: 每列轨道的堆塔数量, height: 每个堆塔的可堆放层数
        """
        self.shape = shape
        self.env = env
        self.total_place = np.prod(shape)
        self.s: np.ndarray = np.zeros(shape, dtype=np.uint8)
        # 为每个堆塔提供一个锁, 一般锁位于每个堆塔顶层货物的下一层, 但当顶层被确定出入货时, 锁在顶层+1层
        self.sync_ = np.ones(shape[:2], dtype=np.bool)
        self.storage_policy = StoragePolicy(store_policy, self)
        logging.info(f"Stocks initializing...\nshape:{self.shape}, occupied rate: {self.s_rate:.2%}, avail stacks:{np.sum(self.sync_)}\nStocks details (2d gridview):\n{self.s_2d}")
        self.debug = {'forced-put': 0}

    @property
    def s_2d(self) -> np.ndarray:
        return np.count_nonzero(self.s, axis=2)

    @property
    def s_1d(self) -> np.ndarray:
        return self.s_2d.sum(axis=1)

    @property
    def value_counts(self):
        z: pd.Series = pd.Series(self.s.ravel()[1:]).value_counts().sort_index()
        return z.values, z.index

    @property
    def s_rate(self):
        return np.count_nonzero(self.s) / self.total_place

    @property
    def s_state(self):
        """
        呈现库存锁定状况
        :return: 返回所有的锁定位置以及锁定堆塔数量
        """
        locked = np.argwhere(self.s_2d != self.sync_)
        return locked, locked.shape[0]

    # @timeit
    def tier_x(self, z):
        """
        从俯视图角度，给出顶层sku内容
        :param z:
        :return: 二维
        """
        non_zero_xyz = np.asarray(self.s.nonzero()).T
        res = np.zeros((NUM_OF_COLS, STACKS_OF_ONE_COL), dtype=np.int)
        for x in range(NUM_OF_COLS):
            for y in range(STACKS_OF_ONE_COL):
                non_zero = np.nonzero(self.s[x][y])[0]
                sku_id = self.s[x, y, non_zero[z-1]] if non_zero.size > z+1 else 0
                res[x, y] = sku_id
        return res

    def rand_S_place(self) -> Tuple[int, int, int]:
        """
        随机一个
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

    def S(self, x, y, sku_id, z=None, is_S=True):
        """
        判断合理性, 入库修改库存
        :param x
        :param y
        :param sku_id: sku id that stored before
        :param z: tier of a storage place, always omitted
        :param is_S: is a storage action, it will be a reshuffling action.
        """
        if z is not None:
            self.s[x, y, z] = sku_id
        elif z is None:
            try:
                z = np.argwhere(self.s[x, y] == 0)[0, 0]
                self.s[x, y, z] = sku_id
            except IndexError:
                logging.log(55, f"{x, y} is forced 'S' a sku-{sku_id}")
                self.s[x, y, NUM_OF_TIERS - 1] = sku_id
                self.debug['forced-put'] += 1
        if is_S:  # recognize the action is reshuffling or not
            self.sync_[x, y] = True

    def R(self, x, y, z=None, is_R=True):
        """
        依据取货位置来进行取货, 判断合理性, 修改库存
        :param x
        :param y
        :param z: tier of a retrieval place, always omitted
        :param is_R: is a retrieval action, it will be a reshuffling action.
        """
        if z is None:
            z = int(np.argwhere(self.s[x, y] == 0)[0, 0] - 1)
        self.s[x, y, z] = 0
        if is_R:
            self.sync_[x, y] = True
            # self.sync[x, y] -= 1
        # print(f"Finish-R:{(x, y, z)}, stack: {self.s[x, y]}")
        # print(f"                     locked: {(x, y)}, {self.sync_[x, y]}")


if __name__ == '__main__':
    test = Stocks(shape=(10, 40, 8))
    s = test.rand_S_place()
    r = test.rand_R_place()
    rr = test.rand_R_x()
    print(f"单列库存:\n{test.s_1d}")
    print(f"平面库存:\n{test.s_2d}")
    print(f"随机入库位置: {s}, 库存: {test.s[s]}")
    print(f"随机出库位置: {r}, 库存:{test.s[r[:2]]}")
