# coding=utf-8
# /usr/bin/Python3.6

"""
Author: Xingwei Chen
Email:cxw19@mails.tsinghua.edu.cn
date:2021/6/8 13:30
"""
import logging
import random
import time
from typing import List, Generator, Any, Dict, TYPE_CHECKING

import numpy as np
import pandas as pd
import torch
from scipy import stats
from scipy.stats.stats import DescribeResult
from simpy import Timeout, Environment, Resource, AnyOf

from ORCSRS.Config import *

if TYPE_CHECKING:
    from ORCSRS.ORSRSRL import RL
from Orders import OrderEntry
from Orders.OrderEntry import InboundOrderEntry, OutboundOrderEntry
from Robots.Fleet import Fleet, Fleet_PST
from Robots.PSB import PSB
from Robots.PST import PST
from Strategy.Orders import OrderDesignate, OrderPool
from Strategy.Scheduling import SchedulingPolicy
from Warehouse.Stocks import Stocks
from Warehouse.WorkStation import WorkStation

np.set_printoptions(linewidth=400)
random.seed(43)


class Energy:
    """
    记录所有值得记录的能耗信息，多采用numpy, 增量计算以减少内存占用
    """

    def __init__(self, num_of_psb, num_of_pst):
        self._S: list = []
        self._R: list = []
        self.PSB = {i: 0 for i in range(num_of_psb)}
        self.PST = {i: 0 for i in range(num_of_pst)}

    @property
    def S_ndarray(self) -> np.ndarray:
        return np.asarray(self._S)

    @property
    def R_ndarray(self):
        return np.asarray(self._R)

    @property
    def All_ndarray(self):
        return np.asarray([*self._S, *self._R])

    def record_S(self, ec):
        self._S.append(ec * 3600)  # KJ

    def record_R(self, ec):
        self._R.append(ec * 3600)  # KJ

    def S_info(self) -> DescribeResult:
        return stats.describe(self.S_ndarray)

    def R_info(self) -> DescribeResult:
        return stats.describe(self.R_ndarray)


class PSS(Environment):
    """
    将PSS直接集成在Simpy的离散事件仿真环境中
    """

    def __init__(self, rl=False):
        self._action = 1
        logging.info("PSS Environment initializing......")
        super().__init__()
        self.rl = rl
        self.BP_fleet: Fleet = Fleet(self, NUM_OF_PSBs)  # bp 多机器人 + 3 psts
        self.TC_fleet: Fleet_PST = Fleet_PST(self, PSTs_left + PSTs_right)  # pst 多机器人
        self.work_stations: List[WorkStation] = [WorkStation(self) for _ in range(NUM_OF_COLS)]
        self.stocks: Stocks = Stocks((NUM_OF_COLS, STACKS_OF_ONE_COL, NUM_OF_TIERS), self)
        self.arrive_order_index, self.finish_order_index, self.order_record = self.order_records_init()
        self.EC = Energy(NUM_OF_PSBs, NUM_OF_PSTs)  # 一个记录能耗信息的类
        self.lines = [Resource(self) for _ in range(NUM_OF_COLS)]
        self.order_pool = OrderPool('FIFO')
        self.strategy = {'Designate': OrderDesignate('random', self.stocks, self.BP_fleet),
                         'Storage': self.stocks.storage_policy,
                         'Scheduling': SchedulingPolicy(self)}
        self.bp_fleet_activate = self.event()
        self.avail_bp = self.event().succeed()
        self.OrderPool_status = self.event()
        self._BP_status = self.event()
        self.complete_event = self.event()
        self.order_arrive_event = self.event()
        self.initialize()
        self.order_arrival = {'R': [], 'S': []}
        self.queue = [0, 0]
        self.observation, self.reward, self.done, self.info = None, None, None, None

    def order_records_init(self):
        logging.info(
            f"Order records pool initializing...\nIncluding arrive orders, finish orders, reshuffle tasks of each col, order details and etc.")
        arrive_order_index: Dict[str, int] = {InboundOrderEntry.type_: 0, OutboundOrderEntry.type_: 0}
        finish_order_index: Dict[str, list] = {InboundOrderEntry.type_: [], OutboundOrderEntry.type_: []}  # 已完成的订单编号
        order_record: Dict[str, Dict[str, OrderEntry]] = {InboundOrderEntry.type_: {}, OutboundOrderEntry.type_: {}}
        return arrive_order_index, finish_order_index, order_record

    def initialize(self):
        """
        初始化生成整个仓库环境
        :return:
        """
        # 绑定离散事件生成器
        logging.info(f"DES processes initializing...\n")
        self.process(self.source_retrieval())
        self.process(self.source_storage())
        if self.rl:
            self.process(self.rl_order_release())

    def log_BP_status(self):
        logging.info(f'Sim time: {self.now:>10.2f}, PSB: {[str(bp) for bp in self.BP_fleet.all_PSBs]}')

    @property
    def reshuffle_task(self):
        return pd.Series([ro.reshuffle for ro in self.order_record['R'].values()])

    def OrderPool_update(self):
        self.OrderPool_status.succeed()
        self.OrderPool_status = self.event()

    @property
    def BP_status(self):
        if len(self.BP_fleet) > 0:
            return self.event().succeed()
        else:
            return self._BP_status

    @property
    def state(self) -> np.ndarray:
        """
        返回当前仿真环境内的状态信息, 并转为tensor
        订单信息记录为 (sku_id, 1 为入库单 2为出库单,waiting time)
        :return:
        """
        BP_state = np.zeros(NUM_OF_COLS, dtype=np.int)
        TC_state = np.zeros((2, NUM_OF_COLS), dtype=np.int)
        for x in self.BP_fleet.at_lines.values():
            BP_state[x] = 1
        for x in self.TC_fleet.at_lines.values():
            TC_state[int(x[1]), x[0]] = 1

        fleet_state = np.pad(np.hstack([BP_state.reshape(-1, 1), TC_state.T]), ((0, 0), (0, 1)), constant_values=(0, 0))
        # orders shape: (100, 3)
        orders = np.asarray(
            [(o.sku_id, int(o.type_ == OutboundOrderEntry.type_) + 1, o.has_waiting_since(self.now)) for o in
             self.order_pool.foreXorders(order_pool_size, self.now) if o]) if len(self.order_pool) else np.zeros(
            (100, 3))
        if orders.shape[0] < order_pool_size:
            orders = np.pad(orders, ((0, order_pool_size - orders.shape[0]), (0, 0)), mode='constant',
                            constant_values=(0, 0))
        orders = orders.reshape(
            (NUM_OF_COLS, int(order_pool_size / NUM_OF_COLS * 3)))
        stocks = self.stocks.s.transpose(2, 0, 1)
        state = np.hstack([np.hstack([fleet_state, orders]), np.hstack(stocks)])
        return state.astype(np.float32)

    def rl_order_release(self):
        """
        when more than one BP available and order pool has orders, we will conduct a order release action.
        """
        action = None
        step_idx = 0
        self.action = 6
        while True:
            # 接收出入库完成任务事件, 来与车辆指派process相交互，一旦有完成订单，将会来实施一次订单绑定小车，下方流程
            # 一g旦有车被R释放+订单池内有订单，将会进入该流程
            yield self.BP_status | self.bp_fleet_activate
            yield self.OrderPool_status
            step_idx += 1
            if step_idx < 100:
                continue
            bp = random.choice(self.BP_fleet.all_PSBs)
            # 进入当前位置就是一个重要记录点[完成了一个订单], 记录环境的变化结果作为step结果
            logging.log(45, f"RLTask step-[{step_idx}]")
            new_state = self.state
            # action = self.rl.agent.select_action(new_state)
            # self.action = action
            order_entry, target_place = self.strategy['Scheduling'].select_order(bp)
            order_entry.d = target_place
            self.reward = self.compute_reward(bp, order_entry, target_place)
            self.info = f""
            self.done = abs(SIM_ELAPSE_RL - self.now) / SIM_ELAPSE_RL < 0.01
            bp = yield self.BP_fleet.get_bp_(bp.name)
            logging.log(10, f"{self.now: .3f}, BP{bp.ID} is bound to Order[{order_entry.name}]")
            order = self.retrieve(int(order_entry.name[1:]), sku_id=order_entry.sku_id, bp=bp, order_entry=order_entry,
                                  target=target_place) \
                if order_entry.type_ == OutboundOrderEntry.type_ else self.storage(int(order_entry.name[1:]), sku_id=order_entry.sku_id, bp=bp, order_entry=order_entry, target=target_place)  # 若是入库订单，则采用原先的默认随机方法
            self.process(order)
            logging.log(10, self.stocks.value_counts)

    def source_retrieval(self):
        """
        柏松到达来生成出库订单
        """
        while True:
            self.arrive_order_index[OutboundOrderEntry.type_] += 1
            time_interval_of_order_arrive: float = random.expovariate(ARRIVAL_RATE)
            yield self.timeout(time_interval_of_order_arrive)
            sku_id = self.strategy['Storage'].order_arrive() if NUM_OF_SKU > 1 else 1
            if self.rl:
                outbound_order = OutboundOrderEntry(
                    f'{OutboundOrderEntry.type_}{self.arrive_order_index[OutboundOrderEntry.type_]}', sku_id=sku_id,
                    arrive_time=self.now)
                self.order_pool.append(outbound_order)
                self.OrderPool_update()
            else:
                outbound_order: Generator[Timeout, Any, Any] = self.retrieve(
                    self.arrive_order_index[OutboundOrderEntry.type_], sku_id)
                self.process(outbound_order)

    def source_storage(self):
        """
        柏松到达来生成入库订单
        """
        while True:
            self.arrive_order_index[InboundOrderEntry.type_] += 1
            time_interval_of_order_arrive: float = random.expovariate(ARRIVAL_RATE)
            yield self.timeout(time_interval_of_order_arrive)
            sku_id = self.strategy['Storage'].order_arrive(inbound=True) if NUM_OF_SKU > 1 else 1
            if self.rl:
                inbound_order = InboundOrderEntry(
                    f"{InboundOrderEntry.type_}{self.arrive_order_index[InboundOrderEntry.type_]}", sku_id=sku_id,
                    arrive_time=self.now)
                self.order_pool.append(inbound_order)
                self.OrderPool_update()
            else:
                inbound_order: Generator[Timeout, Any, Any] = self.storage(
                    self.arrive_order_index[InboundOrderEntry.type_], sku_id)
                logging.info(f"{self.now:>10.2f}, S-{self.arrive_order_index[InboundOrderEntry.type_]} arrived.")
                self.process(inbound_order)
                self.OrderPool_update()

    def retrieve(self, outbound_order_idx, sku_id, **kwargs):
        """
        A generator representing the procedure of outbound task.
        :param sku_id: 产生订单时是否给出id号
        :param outbound_order_idx:R 该出库订单编号
        """
        # 一个取货流程一旦下发后，仅在请求有限资源时会发生离散事件点
        bp: PSB
        tc: PST
        self.queue[0] += 1
        self.queue[1] += len(self.BP_fleet.get_queue)
        stock_info, stock_index = self.stocks.value_counts
        logging.debug(f"BP fleet queue length: {len(self.BP_fleet.get_queue)}")
        logging.debug(f"sku quantity:\n{np.vstack([stock_index.values, stock_info]).astype(int)}")
        logging.debug(f"堆塔锁定数{(~self.stocks.sync_).sum()}")
        if np.any(stock_info < 1):
            raise ValueError("sku为0")
        arrive_time = self.now
        if self.rl:  # RL环境
            o = kwargs['order_entry']
            x = kwargs['bp'].place[0]
            bp = kwargs['bp']
            x, y, tier = kwargs['target']
            outbound_order_name = o.name
        else:  # 非RL环境
            outbound_order_name = f"{OutboundOrderEntry.type_}{outbound_order_idx}"
            o = OutboundOrderEntry(name=outbound_order_name, sku_id=sku_id, arrive_time=self.now)
            # x = self.stocks.rand_R_x()
            x = self.stocks.storage_policy.random_retrieve_line(o.sku_id)
            while x == -1:
                sku_id = self.strategy['Storage'].order_arrive()
                o.reset_another_sku(sku_id=sku_id)
                x = self.stocks.storage_policy.random_retrieve_line(o.sku_id)
            if x in self.BP_fleet.at_lines.values():
                bp = yield self.BP_fleet.get_line(x, lambda psb_: psb_.line == x)
            else:
                bp = yield self.BP_fleet.get_line(x, lambda psb_: psb_.idle)
            x, y, tier, has_sku = self.strategy['Storage'].random_retrieve_from_line(x, sku_id)
            while has_sku == 0:
                sku_id = self.strategy['Storage'].order_arrive()
                o.reset_another_sku(sku_id=sku_id)
                x, y, tier, has_sku = self.strategy['Storage'].random_retrieve_from_line(x, sku_id)
        logging.debug(f"{self.now:>10.2f}, bp lines is {self.BP_fleet.at_lines}, resource:{self.BP_fleet.items}")
        logging.info(f"{self.now:>10.2f}, {outbound_order_name} arrived. target x:{x}")
        self.log_BP_status()

        # x, y, tier = self.stocks.rand_R_from_x(x)
        self.order_arrival['R'].append(sku_id)
        o.update(d=(x, y, tier), o=bp.place, start_time=self.now, sku_id=sku_id)  # 能够拿到psb的资源时，开始计时为工作时间
        o.time_line['catch_psb'] = self.now
        self.order_record[o.type_][outbound_order_name] = o
        bp.occupied_by(o)
        if bp.line != x:
            logging.info(f"bp {bp.ID} change line from [{bp.line}] → [{x}]")
            # 需要PST进行转运, 转运完成后可以进行相同的出库操作  PSB.x → [是否在转运区间]  good.x
            tc = yield self.TC_fleet.get_line(request_line_od=[bp.line, x], filter_=lambda pst_: pst_.idle)
            o.time_line['catch_pst'] = self.now
            tc.occupied_by(o)
            joint_y = -1 if tc.side else STACKS_OF_ONE_COL
            # 事件 pst抵达-1位置, psb抵达-1位置, yield相同的行走时间, 释放pst
            yield self.timeout(max(tc.go_to_horizontally(bp.line, False), bp.go_to_horizontally((bp.line, joint_y))))
            o.time_line['tc&bp_joint'] = self.now
            bp.place = tc.place = (bp.line, joint_y)
            ...
            yield self.timeout(tc.take_psb2line(x))
            bp.place = tc.place = (x, joint_y)
            o.time_line['finish_transfer'] = self.now
            self.TC_fleet.put(tc)
            tc.working_time += self.now - o.time_line['catch_pst']
            tc.released()
        # psb已抵达目标轨道, 从当前位置前往
        t_psb2retrieve_point = bp.go_to_horizontally((x, y), is_loaded=False)
        t_reshuffle = 0
        if tier < self.stocks.s_2d[x, y] - 1:  # 非栈顶sku
            o.reshuffle = self.stocks.s_2d[x, y] - tier - 1
            logging.error(f"{x}, {y}, {tier}")
            t_reshuffle = bp.reshuffle_blocking_bin(self, (x, y), tier, sku_id=sku_id)[-1]
        # 栈顶sku / 已翻箱完成
        t_psb2workstation = bp.retrieve2workstation_without_shuffling(self, (x, y), tier)
        t_retrieve = t_psb2retrieve_point + t_reshuffle + t_psb2workstation
        self.stocks.sync_[x, y] = True

        yield self.timeout(t_retrieve)
        bp.place = (bp.line, -1)
        bp.working_time += self.now - o.time_line['catch_psb']
        o.time_line['retrieve_at_this_line'] = self.now
        bp.released()
        o.end_at(self.now)
        self.BP_fleet.put(bp)
        self.bp_fleet_activate = self.event()
        with self.work_stations[x].request() as req_workstation:
            yield req_workstation
            t_pickup = self.work_stations[x].pickup()
            yield self.timeout(t_pickup)
        o.time_line['pickup'] = self.now
        self.finish_order_index[o.type_].append(outbound_order_idx)
        self.EC.record_R(o.all_ec)
        o.end_stack = self.stocks.s[x, y]
        self.complete_event.succeed()
        self.complete_event = self.event()

    def storage(self, inbound_order_idx, sku_id, **kwargs):
        """
        绑定一个存货[入库订单]流程
        :param sku_id: 在生成订单时便产出sku_id
        :param inbound_order_idx: inbound task index
        """
        arrive_time = self.now
        if 'order_entry' in kwargs:
            o = kwargs['order_entry']
            x = kwargs['bp'].place[0]
            bp = kwargs['bp']
            x, y, z = kwargs['target']
        else:
            inbound_order_name = f"{InboundOrderEntry.type_}{inbound_order_idx}"
            # sku_id = self.strategy['Storage'].order_arrive(inbound=True) if NUM_OF_SKU > 1 else 1
            # x, y, z = self.stocks.rand_S_place()
            x = self.strategy['Storage'].store_line(sku_id=sku_id)
            o = InboundOrderEntry(inbound_order_name, arrive_time=self.now)
            if x in self.BP_fleet.at_lines.values():
                bp = yield self.BP_fleet.get_line(x, lambda psb_: psb_.line == x)
            else:
                bp = yield self.BP_fleet.get_line(x, lambda psb_: psb_.idle)
        self.order_record[o.type_][o.name] = o
        self.log_BP_status()
        with self.lines[x].request() as line_lock:
            yield line_lock
            if not self.rl:
                x, y, z = self.strategy['Storage'].store_to_line(sku_id, x)
            o.update(d=(x, y, z), o=(x, -1), arrive_time=arrive_time, start_time=self.now,
                     start_stack=self.stocks.s[x, y].copy(), sku_id=sku_id)
            self.order_arrival['S'].append(sku_id)
            o.time_line['catch_psb'] = self.now
            bp.occupied_by(o)
            if bp.line != x:
                tc = yield self.TC_fleet.get_line([x, bp.line], lambda pst_: pst_.idle)
                o.time_line['catch_pst'] = self.now
                # 事件 pst抵达-1位置, psb抵达-1位置, yield相同的行走时间, 释放pst
                ...
                joint_y = -1 if tc.side else STACKS_OF_ONE_COL
                tc.occupied_by(o)
                t_2line = max(tc.go_to_horizontally(bp.line), bp.go_to_horizontally((bp.line, joint_y)))
                take_time = tc.take_psb2line(x)
                yield self.timeout(t_2line + take_time)
                tc.released()
                tc.target_xyz = bp.target_xyz = (x, joint_y)
                o.time_line['finish_transfer'] = self.now
                logging.debug(f"bp {bp.ID} change line from [{bp.line}] → [{x}]")
                # 需要PST进行转运, 转运完成后可以进行相同的出库操作
                self.TC_fleet.put(tc)
            t_store = bp.store((x, y), self)
            yield self.timeout(t_store)
            bp.target_xyz = (x, y)
            self.stocks.S(x, y, sku_id=sku_id)
        o.time_line['finish_store'] = self.now
        o.end_at(self.now)
        bp.released()
        self.BP_fleet.put(bp)
        self.bp_fleet_activate = self.event()
        self.EC.record_S(o.all_ec)
        self.finish_order_index[o.type_].append(inbound_order_idx)
        self.complete_event.succeed()
        self.complete_event = self.event()

    def __str__(self) -> str:
        return f"Warehouse Config: {NUM_OF_COLS} × {STACKS_OF_ONE_COL}, with {NUM_OF_PSBs} PSBs, {NUM_OF_PSTs} PSTs"

    def bind(self, ele):
        self.rl = ele

    def print_stats(self):
        """
        print all the information of this simulation.
        :return: None
        """
        EC_R, EC_S = self.EC.R_info(), self.EC.S_info()
        reshuffle_record_of_r = np.asarray([list(ro.ec.values()) for ro in self.order_record['R'].values()])
        logging.log(60
                    ,
                    f"Finish rate: R:{len(self.finish_order_index[OutboundOrderEntry.type_]) / self.arrive_order_index[OutboundOrderEntry.type_]:.2%}, {len(self.finish_order_index[InboundOrderEntry.type_]) / self.arrive_order_index[InboundOrderEntry.type_]:.2%}")
        logging.log(60
                    ,
                    f"Current stocks status: stocks_rate: {self.stocks.s_rate:.2%}, lock_rate: {1 - self.stocks.sync_.mean():.2%}")
        logging.log(60
                    ,
                    f"Robots utility\n{[f'{bp.name}:{bp.working_time / SIM_ELAPSE:.2%}' for bp in self.BP_fleet.all_PSBs]}\n{[f'{pst.name}:{pst.working_time / SIM_ELAPSE:.2%}' for pst in self.TC_fleet.all_PSTs]}")
        logging.log(60
                    ,
                    f"Energy Consumption ↓ \nR-jobs: {EC_R.nobs},  avg: {EC_R.mean:.2f} KJ,  min: {EC_R.minmax[0]:<4.2f} KJ, max: {EC_R.minmax[1]:<5.2f} KJ, var: {EC_R.variance:.2f} KJ^2, total: {np.sum(self.EC.R_ndarray):.2f} KJ")

        logging.log(60
                    ,
                    f"S-jobs: {EC_S.nobs},  avg: {EC_S.mean:.2f} KJ, min: {EC_S.minmax[0]:<4.2f} KJ, max: {EC_S.minmax[1]:<5.2f} KJ, var: {EC_S.variance:.2f} KJ^2, total: {np.sum(self.EC.S_ndarray):.2f} KJ\nR: reshuffle:{reshuffle_record_of_r[:, 1].sum() * 3600:.2f} KJ, general: {reshuffle_record_of_r[:, 0].sum() * 3600:.2f} KJ\nTotal EC: {np.sum(self.EC.R_ndarray) + np.sum(self.EC.S_ndarray):.2f} KJ")
        Time_R, Time_S = stats.describe(np.asarray(
            [(i.waiting_time, i.executing_time) for i in list(self.order_record[OutboundOrderEntry.type_].values()) if
             i.end_time > 0])), \
                         stats.describe(np.asarray([(i.waiting_time, i.executing_time) for i in
                                                    list(self.order_record[InboundOrderEntry.type_].values()) if
                                                    i.end_time > 0]))
        logging.log(60
                    ,
                    f"Reshuffle task: {sum([o.reshuffle > 0 for o in self.order_record['R'].values()]) / self.arrive_order_index['R']:.2%}, {np.asarray([o.reshuffle for o in self.order_record['R'].values() if o.reshuffle > 0]).mean():.2f}")
        logging.log(60
                    ,
                    f"Time ↓ \nR: mean waiting time: {Time_R.mean[0]: 4.1f}s, mean working time: {Time_R.mean[1]:4.1f}s, reshuffle_time part: {np.asarray([ro.reshuffle_time / ro.executing_time for ro in self.order_record['R'].values() if ro.reshuffle > 0]).mean():.2%}")
        logging.log(60
                    , f"S: mean waiting time: {Time_S.mean[0]: 4.1f}s, mean working time: {Time_S.mean[1]:4.1f}s")
        logging.log(60, f'Mean queue length: {self.queue[1] / self.queue[0]:.1f}')
        # logging.log(60, f"Tracks balance[Storage]:{pd.Series([o.d[0] for o in self.order_record['R'].values()]).value_counts()}")
        # logging.log(60, f"Tracks balance[Retrieval]:{pd.Series([o.d[0] for o in self.order_record['R'].values()]).value_counts()}")
        logging.log(60, f"TryExcept in Stocks: {self.stocks.debug}")
        # logging.log(60,
        #             f"completed R:{pd.Series([o.sku_id for o in self.order_record['R'].values() if o.end_time > 0]).value_counts()}")
        # logging.log(60,
        #             f"completed S:{pd.Series([o.sku_id for o in self.order_record['S'].values() if o.end_time > 0]).value_counts()}")
        # logging.log(60,
        #             f"not completed S:{pd.Series([o.sku_id for o in self.order_record['S'].values() if o.end_time > 0]).value_counts()}")
        # logging.log(60,
        #             f"not completed S:{pd.Series([o.sku_id for o in self.order_record['S'].values() if o.end_time > 0]).value_counts()}")
        logging.log(60, f"Modified R orders: {len([o for o in self.order_record['R'].values() if o.modified_sku])}")
        # self.reshuffle_task.value_counts().sort_index().plot.bar()
        # print(f"Retrieval orders sku: {pd.Series(self.order_arrival['R']).value_counts()}")
        # print(f"Storage orders sku: {pd.Series(self.order_arrival['S']).value_counts()}")
        if write_in:
            df = pd.DataFrame(columns=result_cols)
            result = pd.Series(index=result_cols)
            result['sim time'] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            result['N_{sku}'] = NUM_OF_SKU
            result['sku strategy'] = MULTI_SKU if NUM_OF_SKU > 1 else ONE_SKU
            result['storage policy'] = store_policy
            result['lambda'] = int(ARRIVAL_RATE * 3600)
            result['shape'] = f"{NUM_OF_COLS}×{STACKS_OF_ONE_COL}×{NUM_OF_TIERS}"
            result['N_{stack}'] = AVAIL_PLACE
            result['N_{BP}'] = NUM_OF_PSBs
            result['N_{TC}'] = NUM_OF_PSTs
            result['R_{lock}'] = f'{1 - self.stocks.sync_.mean():.2%}'
            result['T_w(s)'] = f'{Time_R.mean[0] + Time_S.mean[0]:.1f}'
            result['T_s(s)'] = f'{Time_R.mean[1] + Time_S.mean[1]:.1f}'
            result['RECO(kJ)'] = f'{EC_R.mean:.2f}'
            result['TREC(kJ)'] = f'{np.sum(self.EC.R_ndarray):.2f}'
            result['SCEO(kJ)'] = f'{EC_S.mean:.2f}'
            result['TSEC(kJ)'] = f'{np.sum(self.EC.S_ndarray):.2f}'
            result['TEC(kJ)'] = f'{np.sum(self.EC.R_ndarray) + np.sum(self.EC.S_ndarray):.2f}'
            result['U_{BP}'] = f'{sum([bp.working_time / SIM_ELAPSE for bp in self.BP_fleet.all_PSBs]) / NUM_OF_PSBs: .2%}'
            result[
                'U_{TC}'] = f'{sum([pst.working_time / SIM_ELAPSE for pst in self.TC_fleet.all_PSTs]) / NUM_OF_PSTs: .2%}'
            result[
                'MR_{per}'] = f"{sum([o.reshuffle > 0 for o in self.order_record['R'].values()]) / self.arrive_order_index['R']:.2%}"
            result[
                'MRTiers'] = f"{np.asarray([o.reshuffle for o in self.order_record['R'].values() if o.reshuffle > 0]).mean():.2f}"
            result[
                'Finish rate(R)'] = f'{len(self.finish_order_index[OutboundOrderEntry.type_]) / self.arrive_order_index[OutboundOrderEntry.type_]:.2%}'
            result[
                'Finish rate(S)'] = f'{len(self.finish_order_index[InboundOrderEntry.type_]) / self.arrive_order_index[InboundOrderEntry.type_]:.2%}'
            result['stocks rate'] = f'{self.stocks.s_rate:.2%}'
            result['lock rate'] = f'{1 - self.stocks.sync_.mean():.2%}'
            result['BP utility'] = f"{[f'{bp.name}:{bp.working_time / SIM_ELAPSE:.2%}' for bp in self.BP_fleet.all_PSBs]}"
            result[
                'TC utility'] = f"{[f'{pst.name}:{pst.working_time / SIM_ELAPSE:.2%}' for pst in self.TC_fleet.all_PSTs]}"
            result['R jobs'] = EC_R.nobs
            result['S jobs'] = EC_S.nobs
            result[
                'Reshuffle time part'] = f"{np.asarray([ro.reshuffle_time / ro.executing_time for ro in self.order_record['R'].values() if ro.reshuffle > 0]).mean():.2%}"
            result['queue length'] = f'{self.queue[1] / self.queue[0]:.1f}'
            df = df.append(result, ignore_index=True)
            df.to_csv(result_csv, mode='a', header=False, index=False)

    def compute_reward(self, bp: 'PSB', order: 'OrderEntry', target_place: tuple):
        x0, y0 = bp.place
        x, y, z = target_place
        stack_info = self.stocks.s[x, y]
        reshuffle_tiers = np.count_nonzero(stack_info[z:]) - 1
        weighted_return = travel_length_weight * (abs(x0 - x) + abs(y0 - y)) + reshuffle_tiers_weight * (
            reshuffle_tiers) + change_track_weight * bool(x0 - x)
        logging.log(10, f"{self.now:>10.2f}, {order.name}, reward: {weighted_return}")
        return weighted_return

    @property
    def action(self):
        return self._action

    @action.setter
    def action(self, v):
        self._action = v
        logging.log(40, f"{self.now: >10.2f}, RLTask[D3QN]->action: {v}")


if __name__ == '__main__':
    import os
    import sys

    t0 = time.time()
    sys.path.append(
        os.path.abspath(os.path.join(os.getcwd(), '/Users/cxw/Learn/2_SIGS/GraduateWork/Code/PSS_based_on_simpy')))
    # logging.basicConfig(level=logging.WARNING, format='', filemode='a+', filename=without_change_track_log_file_path)
    # logging.basicConfig(level=logging.WARNING, format='', filemode='a+', filename=change_track_log_file_path)
    logging.basicConfig(level=log_level, format='')
    # logging.disable(logging.CRITICAL)
    pss = PSS()
    pss.run(until=SIM_ELAPSE)

    logging.log(60, f"time: {pss.now:.2f}")
    logging.log(60, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    logging.log(60, f"Sim:{SIM_ELAPSE}s / {SIM_ELAPSE // (3600 * 24)} × 24h")
    logging.log(60,
                f"PSS Configuration({'RLTask activated' if RL_embedded else 'Non-RLTask'})\nStrategy:{'Single-sku' if NUM_OF_SKU == 1 else 'Multi-sku'}, Storage:{store_policy}\narrival rate:{ARRIVAL_RATE * 3600} orders/hour\nShape:{NUM_OF_COLS}×{STACKS_OF_ONE_COL}×{NUM_OF_TIERS}, AVAIL PLACE:{AVAIL_PLACE}\nRobots:{NUM_OF_PSBs} bp, {NUM_OF_PSTs} tc")
    pss.print_stats()
    print(pss.stocks.value_counts)
    t1 = time.time()
    logging.log(60,
                f'cpu time: {t1 - t0:.2f} s\n--------------------------------------------------------------------------')
