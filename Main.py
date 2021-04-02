# coding=utf-8
# /usr/bin/Python3.6

"""
Author: Xingwei Chen
Email:cxw19@mails.tsinghua.edu.cn
date:2021/3/11 14:18
"""
import logging

import simpy

from src.Fleet import Fleet
from src.PST import PST
from src.Warehouse import Warehouse
from src.Workstation import Workstation
from src.CONSTANT import N_WORKSTATION
from src.Simulation import Simulation

main_logger = logging.getLogger(name="Main")

logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(filename)s %(funcName)s [line:%(lineno)d] %(levelname)s %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        filename="Logging.log",
        filemode="a")

# storage_policy = ["dedicated", "shared", "random&zoned"]
storage_policy = "shared"

# reshuffling_policy = ["immediate", "delayed", "without_reshuffling"]
reshuffling_policy = "without_reshuffling"

psb_dwelling_policy = "dwell in the place where the last transport order finished"

env = simpy.Environment()
warehouse = Warehouse()
fleet = Fleet()
pst = PST()
workstations = [Workstation(i) for i in range(N_WORKSTATION)]
workstation_resources = [simpy.Resource(env) for i in range(N_WORKSTATION)]
sim = Simulation(env, warehouse, fleet, pst, workstations, workstation_resources)

simulation_time = 60 * 60 * 200
env.run(simulation_time)

