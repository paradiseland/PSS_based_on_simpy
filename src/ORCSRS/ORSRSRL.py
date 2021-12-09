# coding=utf-8
# /usr/bin/Python3.6

"""
Author: Xingwei Chen
Email:cxw19@mails.tsinghua.edu.cn
date:2021/11/22 14:32
"""
from typing import TYPE_CHECKING

import gym
from elegantrl.agents.AgentDoubleDQN import AgentDoubleDQN

from ORCSRS.Config import STACKS_OF_ONE_COL, NUM_OF_COLS, NUM_OF_TIERS, order_pool_size

if TYPE_CHECKING:
    from ORCSRS.PSS import PSS


class RL(gym.Env):
    metadata = {"render.modes": []}
    reward_range = (-float("inf"), float("inf"))
    spec = None

    def __init__(self, simpyEnv: 'PSS'):
        super().__init__()
        self.simpyEnv = simpyEnv
        self.agent: AgentDoubleDQN = AgentDoubleDQN()
        self.agent.init(state_dim=int(4 + order_pool_size * 3 / NUM_OF_COLS + NUM_OF_TIERS * STACKS_OF_ONE_COL), action_dim=1, if_per_or_gae=True)
        self.observation_space = 0
        self.action_space = 0

    @property
    def state(self):
        return self.simpyEnv.state

    def reset(self):
        """

        :return:
        """
        self.simpyEnv = PSS()

    def step(self, action):
        pass

    def render(self, **kwargs):
        self.simpyEnv.print_cur_status()

    def seed(self, seed=None):
        random = self


class Action:
    """
    Define the action space of scheduling.
    Description: currently, we has a available bp and order pool
                 we want to choose an strategy to
                    1. assign this psb to an order and
                    2. assign a place to the order.
    """

    def __init__(self, env):
        self.strategies = {}
        self.action_space = list(range(len(self.strategies)))
        self.env = env


