# coding=utf-8
# /usr/bin/Python3.6

"""
Author: Xingwei Chen
Email:cxw19@mails.tsinghua.edu.cn
date:2021/3/11 08:29
"""
import logging
import random
from collections import OrderedDict
from typing import Generator, Any, Tuple, List

from simpy import Environment, Process, Event, Timeout
import simpy
import numpy as np

from PSB import PSB
from Stack import Stack
from src.CONSTANT import ARRIVAL_RATE
from src.Fleet import Fleet, NUM_OF_PSBS
from src.PST import PST
from src.Warehouse import Warehouse, NUM_OF_TIERS, HEIGHT_AVAILABLE, STACKS_OF_ONE_COL, COL_OF_WAREHOUSE
from src.Workstation import Workstations, Workstation


class Simulation:
    """
    define the overall simulation class to realize discrete event simulation.
    """

    """
    1. Order arrives at poisson process
    2. System assigns an available and closer robot to the order OR wait in the queue.
    3. PSB robot moves in shortest path
    4. PSB robot fetches the retrieval bin.
        dedicated storage: pick up the top bin.
    ✓✓✓✓shared storage policy: ✓✓✓✓
                if retrieval bin is on peek: pick up
                else: reshuffling
                            ✓✓✓✓ immediate reshuffling ✓✓✓✓
                                 delayed reshuffling
    5. PSB robot transports bin to designated workstation, drop off and pick up a storage bin.
    6. PSB robot transports to storage point. random stack:
                                at any position zoned stacks: got to the zone determined by turnover.
    7. PSB robot drop off the bin on the top of a randomly stack.
    8. If previous retrieval includes a reshuffling, then returning blocking bins to storage rack.
    """

    """
    1. 一轨一车
    2. 多轨一车
    3. sku的扩展，工作台拣选顺序
    """

    def __init__(self,
                 environment: Environment,
                 warehouse: Warehouse,
                 fleet: Fleet,
                 pst: PST,
                 workstations: Workstations):
        self.env: simpy.Environment = environment
        self.warehouse: Warehouse = warehouse
        self.fleet: Fleet = fleet
        self.PST: PST = pst
        self.workstations: Workstations = workstations
        self.SR: Process = self.env.process(self.source())

    @property
    def energy_consumption(self) -> list:
        """
        register the energy consumption of each PSB
        :return: list of consumption
        """
        return []

    @property
    def utility_consumption(self) -> list:
        """
        :return: list of utility
        """
        return []

    @property
    def utility_fleet(self) -> list:
        """
        register the energy consumption of each PSB
        :return:
        """
        return []

    @property
    def turnover_rate(self) -> float:
        """
        update the turnover rate of the warehouse
        :return: turnover rate instantly
        """
        return 0

    def source(self) -> Generator[Timeout, Any, Any]:
        """
        In simulation time, keeping registering [Storage&Retrieval] process
        into the simulation environment.
        :return: None
        """
        index_order: int = 0
        while True:
            index_order += 1
            storage_order: Generator[Timeout, Any, Any] = self.retrieve("Order-{}".format(index_order))
            self.env.process(storage_order)
            time_interval_of_order_arrive: float = random.expovariate(ARRIVAL_RATE)
            yield self.env.timeout(time_interval_of_order_arrive)

    def retrieve(self, outbound_order_name: str) -> Generator[Timeout, Any, Any]:
        """
        A Python generator as a Process Object.
        work flow of retrieval.
        :param
        :parameter outbound_order_name: the index name of this inbound order
        :return: yield a lot of Timeout Event
        """

        env: Environment = self.env
        arrival_time_of_inbound_order: float = env.now
        logging.info("{:10.2f}, {:s}".format(arrival_time_of_inbound_order, outbound_order_name))
        # yield self.env.timeout(0)

        # destination first dimension is the width direction
        # that is the ordinal number of workstation and psb
        # ARRIVAL: designated place

        order_place, stack_tier = self.warehouse.random_a_place_for_retrieval()  # (x, y), tier
        line, target_stack_order = order_place  # target stack order 为该堆塔在这一列里的排序顺序
        this_stack: Stack = self.warehouse.stock_record[line][target_stack_order]
        # Here,  已经对一个同步对象进行出栈了1，需要考虑
        # here, we give a sync record for this stack, which will be popped goods.
        this_stack.pop_from_record_for_sync()
        logging.info("{:10.2f}, {} target:({},{}), tier:{}".format(
                env.now, outbound_order_name, line, target_stack_order, stack_tier))
        # logging.debug("{} current stack size = {}".format(outbound_order_name, this_stack.size))
        # logging.debug("{} target tier: {}".format(outbound_order_name, stack_tier))
        is_current_line_has_a_psb: List[bool] = self.fleet.has_an_idle_PSB_in_line(line)
        if sum(is_current_line_has_a_psb) < 1:
            # don't have an available psb this line.
            psb_in_this_line: PSB = self.fleet.has_an_PSB_in_line(line)
            if psb_in_this_line:
                psb: PSB = psb_in_this_line
            else:
                psb: PSB = self.fleet.get_an_available_psb_nearly(line)
                req_psb_0 = psb.resource.request()
                #  TODO:
                time_of_psb_came_to_changeTrack_line = psb.go_to_horizontally()
                with self.PST.resource.request() as req_pst:
                    # TODO:  要等待psb到达换轨位置
                    self.PST.occupied()
                    self.PST.move_to_target_line(psb.line)
                    self.PST.transport_PSB_to_target_line()
                    self.PST.released()
                psb.released()
        else:
            psb: PSB = self.fleet.psbs_record[is_current_line_has_a_psb.index(True)]

        logging.info("{:10.2f}, line [{}] has an available psb".format(env.now, line))

        with psb.resource.request() as req_psb:
            yield req_psb
            logging.info("{:10.2f}, {} has seized the [psb_{}]".format(env.now, outbound_order_name, psb.name))
            # 是否为栈顶料箱
            if this_stack.is_peek(stack_tier):
                logging.info("{:10.2f}, target {} is on the peek of that stack. peek:{}".format(
                        env.now, outbound_order_name, this_stack.size_of_record_for_sync))
                logging.info("{:10.2f}, target stack record: {}".format(env.now, stack_tier))
                time_psb2retrieve_point = psb.go_to_horizontally(order_place)
                yield env.timeout(time_psb2retrieve_point)
                psb.update_dwell_point(order_place)
                this_stack.stack.pop()
                logging.info("{:10.2f}, [psb_{}] has arrived at the target place.".format(
                        env.now, target_stack_order))
                time_retrieve2workstation = psb.get_time_retrieve_bin_without_shuffle(stack_tier, target_stack_order)
                yield env.timeout(time_retrieve2workstation)
                psb.update_dwell_point((line, -1))  # update the psb place above the workstation
                logging.info("{:10.2f}, [psb_{}] has transported {} at the target place.".format(
                        env.now, line, outbound_order_name))
            else:  # need to reshuffle this stack
                logging.info("{:10.2f}, target {} is not on the peek of the stack. peek:{}".format(
                        env.now, outbound_order_name, this_stack.size - 1))
                logging.info(
                        "{:10.2f}, target stack stock record: {}".format(
                                env.now, this_stack.stack))
                time_psb2retrieve_point = psb.go_to_horizontally(order_place)
                yield env.timeout(time_psb2retrieve_point)
                psb.update_dwell_point(order_place)
                logging.info(
                        "{:10.2f}, [psb_{}] has arrived at the retrieve point {} for {}.".format(
                                env.now, psb.name, order_place, outbound_order_name))
                # logging.debug("{} current stack size :{}".format(name, warehouse_record[current_line][target_y].size()))
                # logging.debug("{} target tier : {}".format(name, stack_tier))
                if this_stack.size > stack_tier:
                    pass
                else:
                    # FIXME: DEBUG HERE
                    try:
                        stack_tier = np.random.choice(this_stack.size())
                    except ValueError:
                        print('the stack size is less than 0')
                time_reshuffle_return_go = psb.reshuffle_and_get_bin(self.warehouse, order_place, stack_tier,
                                                                     "immediate")
                yield env.timeout(time_reshuffle_return_go)
                logging.info(
                        "{:10.2f}, {} [psb_{}] has finished the reshuffling and went to the workstation.".format(
                                env.now, outbound_order_name, line))
            psb.update_dwell_point((line, -1))
            psb.released()
        with self.workstations.workstations[line].resource.request() as req_workstation:
            logging.info(
                    "{:10.2f}, {} request the workstation".format(
                            env.now, outbound_order_name))
            yield req_workstation
            logging.info(
                    "{:10.2f}, {} seized the workstation".format(
                            env.now, outbound_order_name))
            time_pickup = Workstation.get_time_of_pick_up()
            yield env.timeout(time_pickup)
            logging.info(
                    "{:10.2f}, workstation {} has finished {} picking up".format(
                            env.now, line, outbound_order_name))

        env.process(self.store(line, outbound_order_name))

    def store(self, line: int, inbound_order_name: str):
        # time_storage_arrive = env.now
        env = self.env
        yield env.timeout(0)
        logging.info(
                "{:10.2f}, {}_store has arrived".format(
                        env.now, inbound_order_name))
        choose_height = NUM_OF_TIERS
        target_y = 0
        while choose_height >= HEIGHT_AVAILABLE:
            target_y = random.randint(0, STACKS_OF_ONE_COL - 1)
            choose_height = self.warehouse.stock_record[line][target_y].size_of_record_for_sync

        logging.info(
                "{:10.2f}, {}_store will be stored at {}".format(
                        env.now, inbound_order_name, (line, target_y)))

        self.warehouse.stock_record[line][target_y].push_into_record_for_sync(1)

        if sum(self.fleet.has_an_idle_PSB_in_line(line)) >= 1:  # TODO:
            logging.info(
                    "{:10.2f}, line {} has a psb".format(
                            env.now, line))
            psb = self.fleet.has_an_PSB_in_line(line)
            with psb.resource.request() as req_psb:
                yield req_psb
                logging.info(
                        "{:10.2f}, {}_store seized psb_{}".format(
                                env.now, inbound_order_name, line))

                # label_psb_start = env.now

                time_store = psb.store((line, target_y), self.warehouse)
                self.warehouse.stock_record[line][target_y].push(1)
                yield env.timeout(time_store)
                logging.info(
                        "{:10.2f}, {}_store has been finished".format(
                                env.now, inbound_order_name))
                psb.released()
                psb.update_dwell_point((line, target_y))
                logging.info(
                        "{:10.2f}, [psb_{}]has been released at {}".format(
                                env.now, line, (line, target_y)))
        else:
            pass


def Main():
    logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
            datefmt='%a, %d %b %Y %H:%M:%S',
            filename='Result.log',
            filemode='w')

    # storage_policy = ["dedicated", "shared", "random&zoned"]
    storage_policy = "shared"
    # reshuffling_policy = ["immediate", "delayed"]
    reshuffling_policy = "immediate"
    psb_dwell_policy = "dwell in the place where the last ordered finished"
    constraint_psb = 3

    env = simpy.Environment()
    warehouse = Warehouse()

    psbs_fleet = OrderedDict({i: PSB(i, env, i) for i in range(NUM_OF_PSBS)})
    fleet = Fleet(psbs_fleet)
    pst = PST(env)
    workstations = Workstations(OrderedDict({i: Workstation(i, env) for i in range(COL_OF_WAREHOUSE)}))

    sim = Simulation(env, warehouse, fleet, pst, workstations)
    # simulation_time = 60 * 60 * 8 * 22 * 12
    simulation_time = 60 * 60 * 200
    env.run(simulation_time)


if __name__ == '__main__':
    Main()
