# coding=utf-8
# /usr/bin/Python3.6

"""
Author: Xingwei Chen
Email:cxw19@mails.tsinghua.edu.cn
date:2022/2/9 11:54
"""

import sys, os
import time

from ORCSRS.ORSRSRL import RL
from ORCSRS.Config import *

curr_path = os.path.dirname(os.path.abspath(__file__))  # 当前文件所在绝对路径
parent_path = os.path.dirname(curr_path)  # 父路径
sys.path.append(parent_path)  # 添加路径到系统路径

import gym
import torch
import datetime

from util.utils import save_results, make_dir
from util.utils import plot_rewards
from ddqn import DoubleDQN

curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  # 获取当前时间


class Config:
    def __init__(self):
        ################################## 环境超参数 ###################################
        self.algo_name = 'DoubleDQN'  # 算法名称
        self.env_name = 'ORCSRSRL'  # 环境名称
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")  # 检测GPU
        self.train_eps = 200  # 训练的回合数
        self.test_eps = 30  # 测试的回合数
        ################################################################################

        ################################## 算法超参数 ###################################
        self.gamma = 0.95  # 强化学习中的折扣因子
        self.epsilon_start = 0.95  # e-greedy策略中初始epsilon
        self.epsilon_end = 0.01  # e-greedy策略中的终止epsilon
        self.epsilon_decay = 500  # e-greedy策略中epsilon的衰减率
        self.lr = 0.0001  # 学习率
        self.memory_capacity = 100000  # 经验回放的容量
        self.batch_size = 64  # mini-batch SGD中的批量大小
        self.target_update = 2  # 目标网络的更新频率
        self.hidden_dim = 256  # 网络隐藏层
        ################################################################################

        ################################# 保存结果相关参数 ##############################
        self.result_path = curr_path + "/outputs/" + self.env_name + \
                           '/' + curr_time + '/results/'  # 保存结果的路径
        self.model_path = curr_path + "/outputs/" + self.env_name + \
                          '/' + curr_time + '/models/'  # 保存模型的路径
        self.save = True  # 是否保存图片
        ################################################################################


def env_agent_config(cfg, seed=1):
    # env = gym.make(cfg.env_name)
    env = RL()
    env.seed(seed)
    state_dim = STATE_DIM
    action_dim = ACTION_DIM
    agent = DoubleDQN(state_dim, action_dim, cfg)
    return env, agent


def train(cfg, env, agent):
    print('Start training!')
    print(f'env：{cfg.env_name}, algorithm：{cfg.algo_name}, device：{cfg.device}.')
    rewards = []  # 记录所有回合的奖励
    ma_rewards = []  # 记录所有回合的滑动平均奖励
    for i_ep in range(cfg.train_eps):
        t0 = time.time()
        ep_reward = 0  # 记录一回合内的奖励
        state = env.reset()  # 重置环境，返回初始状态
        while True:
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            ep_reward += reward
            agent.memory.push(state, action, reward, next_state, done)
            state = next_state
            agent.update()
            if done:
                break
        if i_ep % cfg.target_update == 0:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())
        if (i_ep + 1) % 10 == 0:
            print(f'回合：{i_ep + 1}/{cfg.train_eps}，奖励：{ep_reward}')
        rewards.append(ep_reward)
        if ma_rewards:
            ma_rewards.append(
                0.9 * ma_rewards[-1] + 0.1 * ep_reward)
        else:
            ma_rewards.append(ep_reward)
        t1 = time.time()
        print(f"{'*' * 40}\n{i_ep + 1}_th episode:")
        env.print_stats()
        logging.log(60, f'cpu time: {t1 - t0:.1f}s')
    print('Training finished！')
    env.close()
    return rewards, ma_rewards


def test(cfg, env, agent):
    print('开始测试!')
    print(f'环境：{cfg.env_name}, 算法：{cfg.algo_name}, 设备：{cfg.device}')
    ############# 由于测试不需要使用epsilon-greedy策略，所以相应的值设置为0 ###############
    cfg.epsilon_start = 0.0  # e-greedy策略中初始epsilon
    cfg.epsilon_end = 0.0  # e-greedy策略中的终止epsilon
    ################################################################################
    rewards = []  # 记录所有回合的奖励
    ma_rewards = []  # 记录所有回合的滑动平均奖励

    for i_ep in range(cfg.test_eps):
        state = env.reset()
        ep_reward = 0
        while True:
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            state = next_state
            ep_reward += reward
            if done:
                break
        rewards.append(ep_reward)
        if ma_rewards:
            ma_rewards.append(ma_rewards[-1] * 0.9 + ep_reward * 0.1)
        else:
            ma_rewards.append(ep_reward)
        print(f"回合：{i_ep + 1}/{cfg.test_eps}，奖励：{ep_reward:.1f}")
    print('完成测试！')
    env.close()
    return rewards, ma_rewards


if __name__ == "__main__":
    logging.basicConfig(level=50, format='')
    cfg = Config()
    # 训练
    env, agent = env_agent_config(cfg)
    rewards, ma_rewards = train(cfg, env, agent)
    make_dir(cfg.result_path, cfg.model_path)  # 创建保存结果和模型路径的文件夹
    agent.save(path=cfg.model_path)  # 保存模型
    save_results(rewards, ma_rewards, tag='train',
                 path=cfg.result_path)  # 保存结果
    plot_rewards(rewards, ma_rewards, cfg, tag="train")  # 画出结果
    # 测试
    env, agent = env_agent_config(cfg)
    agent.load(path=cfg.model_path)  # 导入模型
    rewards, ma_rewards = test(cfg, env, agent)
    save_results(rewards, ma_rewards, tag='test',
                 path=cfg.result_path)  # 保存结果
    plot_rewards(rewards, ma_rewards, cfg, tag="test")  # 画出结果
