# coding=utf-8
# /usr/bin/Python3.6

"""
Author: Xingwei Chen
Email:cxw19@mails.tsinghua.edu.cn
date:2021/11/22 14:32
"""
import random
import time
from collections import Counter
from typing import TYPE_CHECKING
import logging
import gym
import numpy as np
import torch
from elegantrl.agents.AgentDoubleDQN import AgentD3QN
from elegantrl.train.config import Arguments
from elegantrl.train.run import train_and_evaluate
from elegantrl.train.utils import init_agent, init_evaluator, init_replay_buffer
from simpy import AnyOf, Environment, AllOf

from ORCSRS.Config import STACKS_OF_ONE_COL, NUM_OF_COLS, NUM_OF_TIERS, order_pool_size, SIM_ELAPSE, STATE_DIM, \
    ACTION_DIM, log_level, RL_embedded, RANDOM_SEED
from Orders.OrderEntry import OutboundOrderEntry

from ORCSRS.PSS import PSS


def train(args: Arguments):
    args.init_before_training()
    learned_gpu = args.learner_gpus[0]
    env = RL()
    agent = init_agent(args, gpu_id=learned_gpu, env=env)
    evaluator = init_evaluator(args, agent_id=0)
    buffer, update_buffer = init_replay_buffer(args, learned_gpu, agent, env=env)

    """start training"""
    cwd = args.cwd
    break_step = args.break_step
    batch_size = args.batch_size
    target_step = args.target_step
    repeat_times = args.repeat_times
    if_allow_break = args.if_allow_break
    soft_update_tau = args.soft_update_tau
    del args
    '''start training loop'''
    if_train = True
    torch.set_grad_enabled(False)
    while if_train:
        traj_list = agent.explore_env(env, target_step)
        steps, r_exp = update_buffer(traj_list)

        torch.set_grad_enabled(True)
        logging_tuple = agent.update_net(buffer, batch_size, repeat_times, soft_update_tau)
        torch.set_grad_enabled(False)

        if_reach_goal, if_save = evaluator.evaluate_and_save(agent.act, steps, r_exp, logging_tuple)
        if_train = not ((if_allow_break and if_reach_goal)
                        or evaluator.total_step > break_step
                        or os.path.exists(f'{cwd}/stop'))

    print(f'| UsedTime: {time.time() - evaluator.start_time:>7.0f} | SavedDir: {cwd}')

    agent.save_or_load_agent(cwd, if_save=True)
    buffer.save_or_load_history(cwd, if_save=True) if agent.if_off_policy else None
    evaluator.save_or_load_recoder(if_save=True)


class RL(gym.Env):
    """
    A Reinforcement Learning embedded to a ORCSRS Simpy environment.

    Any environment needs:
    * A state space, statistics from simpy environment
    * A reward structure, weight sum of reshuffle tiers, order priority,
    * An initialize (reset) method that returns the initial observations
    * A choice of actions
    * A way to make sure the action is legal/possible
    * A step method that passes an action to the environment and returns:
        1. the state new observations
        2. reward
        3. whether state is terminal
        4. additional information
    * A method to render the environment
    * A way to recognize and return a terminal state(end of episode)
    """
    metadata = {"render.modes": []}
    reward_range = (-float("inf"), float("inf"))
    spec = None

    def __init__(self):
        super().__init__()
        # self.simpyEnv = simpyEnv
        self.simpyEnv: 'PSS' = Environment()
        self.env_num = 1
        self.env_name = 'PSS-RLTask'
        self.max_step = 200
        self.state_dim = STATE_DIM
        self.action_dim = ACTION_DIM
        self.if_discrete = True
        self.target_return = -200
        self.agent: AgentD3QN = AgentD3QN()
        self.action_record = Counter()

    @property
    def simpy_state(self):
        return self.simpyEnv.state.ravel()

    def reset(self) -> np.ndarray:
        """
        :return: np.ndarray
        """
        self.simpyEnv = PSS(rl=True)
        self.simpyEnv.bind(self)
        # print(self.simpyEnv.state)
        # self.simpyEnv.run(until=10000)
        init_state = self.simpy_state
        self.action_record = Counter()
        return self.simpy_state

    def step(self, action):
        """
        Standard step function for RLTask env.
        :param action:
        :return:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (bool): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """
        self.action_record[int(action)] += 1
        self.simpyEnv.action = action
        self.simpyEnv.run(until=AllOf(self.simpyEnv, [self.simpyEnv.OrderPool_status, self.simpyEnv.complete_event]))
        observation, reward, info, done = self.simpy_state, self.simpyEnv.reward, self.simpyEnv.info, self.simpyEnv.done
        return observation, reward, done, info

    def render(self, **kwargs):
        self.simpyEnv.log_BP_status()

    def seed(self, seed=None):
        random.seed(seed)

    def print_stats(self):
        self.simpyEnv.print_stats()
        logging.log(60, f'action: {self.action_record}')


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
    env = RL()
    agent = AgentD3QN()
    args = Arguments(agent=agent, env=env)
    args.gamma = 0.97
    args.state_dim = STATE_DIM
    args.action_dim = ACTION_DIM
    args.if_discrete = True
    args.random_seed = RANDOM_SEED
    args.worker_num = 1
    args.cwd = 'src/Result/RLModel'
    train(args)
    # logging.log(60, f"time: {pss.now:.2f}")
    # logging.log(60, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    # logging.log(60, f"Sim:{SIM_ELAPSE}s / {SIM_ELAPSE // (3600 * 24)} × 24h")
    # logging.log(60,
    #             f"PSS Configuration({'RLTask activated' if RL_embedded else 'Non-RLTask'})\nStrategy:{'Single-sku' if NUM_OF_SKU == 1 else 'Multi-sku'}, Storage:{store_policy}\narrival rate:{ARRIVAL_RATE * 3600} orders/hour\nShape:{NUM_OF_COLS}×{STACKS_OF_ONE_COL}×{NUM_OF_TIERS}, AVAIL PLACE:{AVAIL_PLACE}\nRobots:{NUM_OF_PSBs} bp, {NUM_OF_PSTs} tc")
    # pss.print_stats()
    # print(pss.stocks.value_counts)
    # t1 = time.time()
    # logging.log(60,
    #             f'cpu time: {t1 - t0:.2f} s\n--------------------------------------------------------------------------')
