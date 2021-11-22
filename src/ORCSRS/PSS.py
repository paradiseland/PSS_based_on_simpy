# coding=utf-8
# /usr/bin/Python3.6

"""
Author: Xingwei Chen
Email:cxw19@mails.tsinghua.edu.cn
date:2021/6/8 13:30
"""
import random
import time
from typing import List, Generator, Any, Dict

import pandas as pd
from scipy import stats
from scipy.stats.stats import DescribeResult
from simpy import Timeout, Environment, Resource

from ORCSRS.Config import *
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
random.seed(42)


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

    def __init__(self):
        logging.info("PSS Environment initializing......")
        super().__init__()
        self.BP_fleet: Fleet = Fleet(self, NUM_OF_PSBs)  # psb 多机器人 + 3 psts
        self.TC_fleet: Fleet_PST = Fleet_PST(self, PSTs_left + PSTs_right)  # pst 多机器人
        self.work_stations: List[WorkStation] = [WorkStation(self) for _ in range(NUM_OF_COLS)]
        self.stocks: Stocks = Stocks((NUM_OF_COLS, STACKS_OF_ONE_COL, NUM_OF_TIERS), self)

        self.arrive_order_index, self.finish_order_index, self.order_record = self.order_records_init()
        self.EC = Energy(NUM_OF_PSBs, NUM_OF_PSTs)  # 一个记录能耗信息的类
        self.lines = [Resource(self) for _ in range(NUM_OF_COLS)]
        self.strategy = {'Designate' : OrderDesignate('random', self.stocks, self.BP_fleet),
                         'Storage'   : self.stocks.storage_policy,
                         'Scheduling': SchedulingPolicy()}
        self.order_pool = OrderPool('FIFO')
        self.initialize()
        self.order_arrival = {'R': [], 'S': []}
        self.queue = [0, 0]
        print(self.stocks.value_counts)

    def order_records_init(self):
        logging.info(f"Order records pool initializing...\nIncluding arrive orders, finish orders, reshuffle tasks of each col, order details and etc.")
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
        # self.process(self.batch_release())  # TODO:  订单池释放订单给机器人

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
                    , f"Current stocks status: stocks_rate: {self.stocks.s_rate:.2%}, lock_rate: {1 - self.stocks.sync_.mean():.2%}")
        logging.log(60
                    ,
                    f"Robots utility\n{[f'{psb.name}:{psb.working_time / SIM_ELAPSE:.2%}' for psb in self.BP_fleet.all_PSBs]}\n{[f'{pst.name}:{pst.working_time / SIM_ELAPSE:.2%}' for pst in self.TC_fleet.all_PSTs]}")
        logging.log(60
                    ,
                    f"Energy Consumption ↓ \nR-jobs: {EC_R.nobs},  avg: {EC_R.mean:.2f} KJ,  min: {EC_R.minmax[0]:<4.2f} KJ, max: {EC_R.minmax[1]:<5.2f} KJ, var: {EC_R.variance:.2f} KJ^2, total: {np.sum(self.EC.R_ndarray):.2f} KJ")

        logging.log(60
                    ,
                    f"S-jobs: {EC_S.nobs},  avg: {EC_S.mean:.2f} KJ, min: {EC_S.minmax[0]:<4.2f} KJ, max: {EC_S.minmax[1]:<5.2f} KJ, var: {EC_S.variance:.2f} KJ^2, total: {np.sum(self.EC.S_ndarray):.2f} KJ\nR: reshuffle:{reshuffle_record_of_r[:, 1].sum() * 3600:.2f} KJ, general: {reshuffle_record_of_r[:, 0].sum() * 3600:.2f} KJ\nTotal EC: {np.sum(self.EC.R_ndarray) + np.sum(self.EC.S_ndarray):.2f} KJ")
        Time_R, Time_S = stats.describe(np.asarray([(i.waiting_time, i.executing_time) for i in list(self.order_record[OutboundOrderEntry.type_].values()) if i.end_time > 0])), \
                         stats.describe(np.asarray([(i.waiting_time, i.executing_time) for i in list(self.order_record[InboundOrderEntry.type_].values()) if i.end_time > 0]))
        logging.log(60
                    ,
                    f"Reshuffle task: {sum([o.reshuffle > 0 for o in self.order_record['R'].values()]) / self.arrive_order_index['R']:.2%}, {np.asarray([o.reshuffle for o in self.order_record['R'].values() if o.reshuffle > 0]).mean():.2f}")
        logging.log(60
                    ,
                    f"Time ↓ \nR: mean waiting time: {Time_R.mean[0]: 4.1f}s, mean working time: {Time_R.mean[1]:4.1f}s, reshuffle_time part: {np.asarray([ro.reshuffle_time / ro.executing_time for ro in self.order_record['R'].values() if ro.reshuffle > 0]).mean():.2%}")
        logging.log(60
                    , f"S: mean waiting time: {Time_S.mean[0]: 4.1f}s, mean working time: {Time_S.mean[1]:4.1f}s")
        logging.log(60, f'Mean queue length: {self.queue[1] / self.queue[0]:.1f}')
        self.reshuffle_task.value_counts().sort_index().plot.bar()
        # print(f"Retrieval orders sku: {pd.Series(self.order_arrival['R']).value_counts()}")
        # print(f"Storage orders sku: {pd.Series(self.order_arrival['S']).value_counts()}")
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
        result['U_{BP}'] = f'{sum([psb.working_time / SIM_ELAPSE for psb in self.BP_fleet.all_PSBs]) / NUM_OF_PSBs: .2%}'
        result['U_{TC}'] = f'{sum([pst.working_time / SIM_ELAPSE for pst in self.TC_fleet.all_PSTs]) / NUM_OF_PSTs: .2%}'
        result['MR_{per}'] = f"{sum([o.reshuffle > 0 for o in self.order_record['R'].values()]) / self.arrive_order_index['R']:.2%}"
        result['MRTiers'] = f"{np.asarray([o.reshuffle for o in self.order_record['R'].values() if o.reshuffle > 0]).mean():.2f}"
        result['Finish rate(R)'] = f'{len(self.finish_order_index[OutboundOrderEntry.type_]) / self.arrive_order_index[OutboundOrderEntry.type_]:.2%}'
        result['Finish rate(S)'] = f'{len(self.finish_order_index[InboundOrderEntry.type_]) / self.arrive_order_index[InboundOrderEntry.type_]:.2%}'
        result['stocks rate'] = f'{self.stocks.s_rate:.2%}'
        result['lock rate'] = f'{1 - self.stocks.sync_.mean():.2%}'
        result['BP utility'] = f"{[f'{psb.name}:{psb.working_time / SIM_ELAPSE:.2%}' for psb in self.BP_fleet.all_PSBs]}"
        result['TC utility'] = f"{[f'{pst.name}:{pst.working_time / SIM_ELAPSE:.2%}' for pst in self.TC_fleet.all_PSTs]}"
        result['R jobs'] = EC_R.nobs
        result['S jobs'] = EC_S.nobs
        result['Reshuffle time part'] = f"{np.asarray([ro.reshuffle_time / ro.executing_time for ro in self.order_record['R'].values() if ro.reshuffle > 0]).mean():.2%}"
        result['queue length'] = f'{self.queue[1] / self.queue[0]:.1f}'
        df = df.append(result, ignore_index=True)
        if write_in:
            df.to_csv(result_csv, mode='a', header=False, index=False)

    def print_cur_status(self):
        logging.info(f'Sim time: {self.now:>10.2f}, PSB: {[str(psb) for psb in self.BP_fleet.all_PSBs]}')

    @property
    def reshuffle_task(self):
        return pd.Series([ro.reshuffle for ro in self.order_record['R'].values()])

    def robots_find_order(self):
        # 改为车找订单

        pass

    def source_retrieval(self):
        """
        柏松到达来生成出库订单
        """
        while True:
            self.arrive_order_index[OutboundOrderEntry.type_] += 1
            time_interval_of_order_arrive: float = random.expovariate(ARRIVAL_RATE)
            yield self.timeout(time_interval_of_order_arrive)
            outbound_order: Generator[Timeout, Any, Any] = self.retrieve(self.arrive_order_index[OutboundOrderEntry.type_])
            self.process(outbound_order)

    def source_storage(self):
        """
        柏松到达来生成入库订单
        """
        while True:
            self.arrive_order_index[InboundOrderEntry.type_] += 1
            time_interval_of_order_arrive: float = random.expovariate(ARRIVAL_RATE)
            yield self.timeout(time_interval_of_order_arrive)
            inbound_order: Generator[Timeout, Any, Any] = self.storage(self.arrive_order_index[InboundOrderEntry.type_])
            logging.info(f"{self.now:>10.2f}, S-{self.arrive_order_index[InboundOrderEntry.type_]} arrived.")
            self.process(inbound_order)

    def retrieve(self, outbound_order_idx):
        """
        A generator representing the procedure of outbound task.
        :param outbound_order_idx:R 该出库订单编号
        """
        # 一个取货流程一旦下发后，仅在请求有限资源时会发生离散事件点
        psb: PSB
        pst: PST
        stock_info = pd.Series(self.stocks.s.ravel()[1:]).value_counts().sort_index()[1:].to_numpy()
        if np.any(stock_info < 1):
            raise ValueError("sku为0")
        logging.debug(f"Filter Store queue: {len(self.BP_fleet.get_queue)}")
        self.queue[0] += 1
        self.queue[1] += len(self.BP_fleet.get_queue)
        logging.debug(f"sku:{stock_info}")
        logging.debug(f"堆塔锁定数{(~self.stocks.sync_).sum()}")
        outbound_order_name = f"{OutboundOrderEntry.type_}{outbound_order_idx}"
        sku_id = self.strategy['Storage'].order_arrive() if NUM_OF_SKU > 1 else 1
        o = OutboundOrderEntry(name=outbound_order_name, sku_id=sku_id, arrive_time=self.now)
        # x = self.stocks.rand_R_x()
        x = self.stocks.storage_policy.random_retrieve_line(o.sku_id)

        # print(f"{self.now:>1D0.2f}, {outbound_order_name}#({x:02d}-{y:02d}-{tier:1d}) arrived.")
        # 申请一台psb, 如果没有, 则会一直等待, 否则将申请附近最近可用车辆
        logging.debug(f"{self.now:>10.2f}, psb lines is {self.BP_fleet.at_lines}, resource:{self.BP_fleet.items}")
        logging.info(f"{self.now:>10.2f}, {outbound_order_name} arrived. target x:{x}")
        self.print_cur_status()
        if x in self.BP_fleet.at_lines.values():
            psb = yield self.BP_fleet.get_line(x, lambda psb_: psb_.line == x)
        else:
            psb = yield self.BP_fleet.get_line(x, lambda psb_: psb_.idle)
        # x, y, tier = self.stocks.rand_R_from_x(x)
        x, y, tier, has_sku = self.strategy['Storage'].random_retrieve_from_line(x, sku_id)
        self.order_arrival['R'].append(sku_id)
        if not has_sku:
            psb.released()
            self.order_pool.append(o)
            return
        o.update(d=(x, y, tier), o=psb.place, start_time=self.now, sku_id=sku_id)  # 能够拿到psb的资源时，开始计时为工作时间
        o.time_line['catch_psb'] = self.now
        self.order_record[o.type_][outbound_order_name] = o
        psb.occupied_by(o)
        if psb.line != x:
            logging.info(f"psb {psb.ID} change line from [{psb.line}] → [{x}]")
            # 需要PST进行转运, 转运完成后可以进行相同的出库操作  PSB.x → [是否在转运区间]  good.x
            pst = yield self.TC_fleet.get_line(request_line_od=[psb.line, x], filter_=lambda pst_: pst_.idle)
            o.time_line['catch_pst'] = self.now
            # pst = fleet_pst.valuex
            pst.occupied_by(o)
            joint_y = -1 if pst.side else STACKS_OF_ONE_COL
            # 事件 pst抵达-1位置, psb抵达-1位置, yield相同的行走时间, 释放pst
            yield self.timeout(max(pst.go_to_horizontally(psb.line, False), psb.go_to_horizontally((psb.line, joint_y))))
            o.time_line['pst&psb_joint'] = self.now
            psb.place = pst.place = (psb.line, joint_y)
            ...
            yield self.timeout(pst.take_psb2line(x))
            psb.place = pst.place = (x, joint_y)
            o.time_line['finish_transfer'] = self.now
            self.TC_fleet.put(pst)
            pst.working_time += self.now - o.time_line['catch_pst']
            pst.released()
        # psb已抵达目标轨道, 从当前位置前往
        t_psb2retrieve_point = psb.go_to_horizontally((x, y), is_loaded=False)
        t_reshuffle = 0
        if tier < self.stocks.s_2d[x, y] - 1:  # 非栈顶sku
            o.reshuffle = self.stocks.s_2d[x, y] - tier - 1
            logging.error(f"{x}, {y}, {tier}")
            t_reshuffle = psb.reshuffle_blocking_bin(self, (x, y), tier, sku_id=sku_id)[-1]
        # 栈顶sku / 已翻箱完成
        t_psb2workstation = psb.retrieve2workstation_without_shuffling(self, (x, y), tier)
        t_retrieve = t_psb2retrieve_point + t_reshuffle + t_psb2workstation
        self.stocks.sync_[x, y] = True

        yield self.timeout(t_retrieve)
        psb.place = (psb.line, -1)
        psb.working_time += self.now - o.time_line['catch_psb']
        o.time_line['retrieve_at_this_line'] = self.now
        psb.released()
        o.end_at(self.now)
        self.BP_fleet.put(psb)
        with self.work_stations[x].request() as req_workstation:
            yield req_workstation
            t_pickup = self.work_stations[x].pickup()
            yield self.timeout(t_pickup)
        o.time_line['pickup'] = self.now
        self.finish_order_index[o.type_].append(outbound_order_idx)
        self.EC.record_R(o.all_ec)
        o.end_stack = self.stocks.s[x, y]
        # TODO:  这儿存在机器人释放信号，直接绑定下一次的订单
        # if self.order_pool.release_policy == 'FIFO':
        #     next_order = self.order_pool.pop()
        #     self.process(next_order)
        # 如 env.process(self.order_pool)

    def storage(self, inbound_order_idx):
        """
        绑定一个存货[入库订单]流程
        :param inbound_order_idx: inbound task index
        """
        inbound_order_name = f"{InboundOrderEntry.type_}{inbound_order_idx}"
        sku_id = self.strategy['Storage'].order_arrive(inbound=True) if NUM_OF_SKU > 1 else 1
        # x, y, z = self.stocks.rand_S_place()
        x = self.strategy['Storage'].store_line(sku_id=sku_id)
        o = InboundOrderEntry(inbound_order_name, arrive_time=self.now)
        self.order_record[o.type_][o.name] = o
        arrive_time = self.now
        self.print_cur_status()

        if x in self.BP_fleet.at_lines.values():
            psb = yield self.BP_fleet.get_line(x, lambda psb_: psb_.line == x)
        else:
            psb = yield self.BP_fleet.get_line(x, lambda psb_: psb_.idle)
        with self.lines[x].request() as line_lock:
            yield line_lock
            x, y, z = self.strategy['Storage'].store_to_line(sku_id, x)
            o.update(d=(x, y, z), o=(x, -1), arrive_time=arrive_time, start_time=self.now, start_stack=self.stocks.s[x, y].copy(), sku_id=sku_id)
            self.order_arrival['S'].append(sku_id)
            o.time_line['catch_psb'] = self.now
            psb.occupied_by(o)
            if psb.line != x:
                pst = yield self.TC_fleet.get_line([x, psb.line], lambda pst_: pst_.idle)
                o.time_line['catch_pst'] = self.now
                # 事件 pst抵达-1位置, psb抵达-1位置, yield相同的行走时间, 释放pst
                ...
                joint_y = -1 if pst.side else STACKS_OF_ONE_COL
                pst.occupied_by(o)
                t_2line = max(pst.go_to_horizontally(psb.line), psb.go_to_horizontally((psb.line, joint_y)))
                take_time = pst.take_psb2line(x)
                yield self.timeout(t_2line + take_time)
                pst.released()
                pst.target_xyz = psb.target_xyz = (x, joint_y)
                o.time_line['finish_transfer'] = self.now
                logging.debug(f"psb {psb.ID} change line from [{psb.line}] → [{x}]")
                # 需要PST进行转运, 转运完成后可以进行相同的出库操作
                self.TC_fleet.put(pst)
            t_store = psb.store((x, y), self)
            yield self.timeout(t_store)
            psb.target_xyz = (x, y)
            self.stocks.S(x, y, sku_id=sku_id)
        o.time_line['finish_store'] = self.now
        o.end_at(self.now)
        psb.released()
        self.BP_fleet.put(psb)
        self.EC.record_S(o.all_ec)
        self.finish_order_index[o.type_].append(inbound_order_idx)

    def __str__(self) -> str:
        return f"{NUM_OF_COLS} × {STACKS_OF_ONE_COL}, with {NUM_OF_PSBs} PSBs, {NUM_OF_PSTs} PSTs"


def grid_search(total_place):
    shapes = []
    tiers = list(range(5, 11))
    for i in tiers:
        s2d = total_place // i
        root = round(s2d ** .5)
        lower = s2d // 110
        upper = s2d // 11
        for j in range(lower, upper, 5):
            k = s2d // j
            shapes.append([i, j, k])
    return shapes


def main(para):
    pass


if __name__ == '__main__':
    import os
    import sys

    t0 = time.time()
    sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '/Users/cxw/Learn/2_SIGS/GraduateWork/Code/PSS_based_on_simpy')))
    # logging.basicConfig(level=logging.WARNING, format='', filemode='a+', filename=without_change_track_log_file_path)
    # logging.basicConfig(level=logging.WARNING, format='', filemode='a+', filename=change_track_log_file_path)
    logging.basicConfig(level=log_level, format='')
    # logging.disable(logging.CRITICAL)
    pss = PSS()
    pss.run(until=SIM_ELAPSE)
    logging.log(60, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    logging.log(60, f"Sim:{SIM_ELAPSE}s / {SIM_ELAPSE // (3600 * 24)} × 24h")
    logging.log(60,
                f"PSS Configuration\nStrategy:{'Single-sku' if NUM_OF_SKU == 1 else 'Multi-sku'}, Storage:{store_policy}\narrival rate:{ARRIVAL_RATE * 3600} orders/hour\nShape:{NUM_OF_COLS}×{STACKS_OF_ONE_COL}×{NUM_OF_TIERS}, AVAIL PLACE:{AVAIL_PLACE}\nRobots:{NUM_OF_PSBs} psb, {NUM_OF_PSTs} pst")
    pss.print_stats()
    print(pss.stocks.value_counts)
    t1 = time.time()
    logging.log(60, f'cpu time: {t1 - t0:.2f} s\n--------------------------------------------------------------------------')
