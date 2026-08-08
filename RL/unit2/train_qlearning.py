"""
Unit 2 作业：从零手写 Q-Learning —— 没有神经网络，就是一张表格。
对应 HF Deep RL Course unit2.ipynb，代码照搬官方 solution，每行加中文讲解。

Q-Learning 的核心思想：
  维护一张 Q 表，Q[状态][动作] = 「在这个状态做这个动作，预期能拿到多少长期回报」。
  agent 不断试错，每走一步就用贝尔曼公式把这张表更新得更准。
  最后在每个状态查表选分最高的动作，就是最优策略。

这就是上次说的「Unit 2 连神经网络都不是，模型就是 500×6 的表格」。

跑法：
    python train_qlearning.py
"""

import numpy as np
import random
import gymnasium as gym
from tqdm import tqdm


# ════════════════════════════════════════════════════════════════════
# 三个核心函数（notebook cell 39 / 44 / 48，官方 solution）
# ════════════════════════════════════════════════════════════════════

def initialize_q_table(state_space, action_space):
    """建一张全 0 的 Q 表，形状 = (状态数, 动作数)。一开始 agent 什么都不知道，全填 0。"""
    return np.zeros((state_space, action_space))


def greedy_policy(Qtable, state):
    """贪心策略（利用 exploitation）：在当前状态，直接选 Q 值最高的动作。评估时用它。"""
    return np.argmax(Qtable[state][:])


def epsilon_greedy_policy(Qtable, state, epsilon):
    """
    ε-贪心策略：训练时用。
    以 (1-ε) 概率选当前最优动作（利用），以 ε 概率随机乱试（探索 exploration）。
    为什么要随机探索？因为一开始的 Q 表是错的，只有多试没试过的动作，才可能发现更好的路。
    """
    if random.uniform(0, 1) > epsilon:
        return greedy_policy(Qtable, state)          # 利用：走已知最好的
    else:
        return env.action_space.sample()             # 探索：随机走一步


# ════════════════════════════════════════════════════════════════════
# 训练循环（notebook cell 54，官方 solution）—— Q-Learning 的心脏
# ════════════════════════════════════════════════════════════════════

def train(n_training_episodes, min_epsilon, max_epsilon, decay_rate, env, max_steps, Qtable):
    for episode in tqdm(range(n_training_episodes)):
        # ε 随训练衰减：初期多探索（ε≈1），后期几乎只利用（ε→0.05）。这行是指数衰减公式。
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)

        state, info = env.reset()                    # 每一局从头开始
        for step in range(max_steps):
            action = epsilon_greedy_policy(Qtable, state, epsilon)   # 选动作
            new_state, reward, terminated, truncated, info = env.step(action)  # 环境反馈

            # ★贝尔曼更新★ 整个算法就这一行：
            #   新Q = 旧Q + 学习率 × [ 即时奖励 + γ×下一状态最优Q − 旧Q ]
            #   中括号里叫「TD 误差」：实际体验到的价值 与 当前表格估计 之差。用它把表格拉得更准。
            Qtable[state][action] = Qtable[state][action] + learning_rate * (
                reward + gamma * np.max(Qtable[new_state]) - Qtable[state][action]
            )

            if terminated or truncated:              # 掉进冰窟/到终点/超时 → 本局结束
                break
            state = new_state                        # 否则进入下一状态，继续
    return Qtable


# ════════════════════════════════════════════════════════════════════
# 评估（notebook cell 60，官方 solution）：关掉探索，纯贪心跑 N 局看平均分
# ════════════════════════════════════════════════════════════════════

def evaluate_agent(env, max_steps, n_eval_episodes, Q, seed):
    episode_rewards = []
    for episode in tqdm(range(n_eval_episodes)):
        # Taxi 用固定 seed 列表，保证每个人的 agent 从同样的起点评测（课程排行榜公平性）
        if seed:
            state, info = env.reset(seed=seed[episode])
        else:
            state, info = env.reset()
        total_rewards_ep = 0
        for step in range(max_steps):
            action = greedy_policy(Q, state)         # 评估只用贪心，不再随机
            new_state, reward, terminated, truncated, info = env.step(action)
            total_rewards_ep += reward
            if terminated or truncated:
                break
            state = new_state
        episode_rewards.append(total_rewards_ep)
    return np.mean(episode_rewards), np.std(episode_rewards)


# ════════════════════════════════════════════════════════════════════
# 环境 1：FrozenLake-v1（notebook cell 26 + 50）—— 4×4 冰湖，从起点走到礼物不掉窟窿
# ════════════════════════════════════════════════════════════════════

def run_frozenlake():
    global env, learning_rate, gamma
    # is_slippery=False = 地不滑，走哪是哪（新手版）。状态空间只有 16 格。
    env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=False)
    state_space, action_space = env.observation_space.n, env.action_space.n
    print(f"\n[FrozenLake] 状态数={state_space}, 动作数={action_space}  → Q 表大小 {state_space}×{action_space}")

    # 超参数（cell 50）
    n_training_episodes, learning_rate, gamma = 10000, 0.7, 0.95
    max_steps, max_epsilon, min_epsilon, decay_rate = 99, 1.0, 0.05, 0.0005

    Qtable = initialize_q_table(state_space, action_space)
    Qtable = train(n_training_episodes, min_epsilon, max_epsilon, decay_rate, env, max_steps, Qtable)
    mean_r, std_r = evaluate_agent(env, max_steps, 100, Qtable, [])
    print(f"[FrozenLake] mean_reward = {mean_r:.2f} +/- {std_r:.2f}  (满分 1.0；每次都能拿到礼物就是 1.0)")


# ════════════════════════════════════════════════════════════════════
# 环境 2：Taxi-v3（notebook cell 87）—— 接客送客，500 个状态，就是上次说的那张 500×6 表
# ════════════════════════════════════════════════════════════════════

def run_taxi():
    global env, learning_rate, gamma
    env = gym.make("Taxi-v3")
    state_space, action_space = env.observation_space.n, env.action_space.n
    print(f"\n[Taxi] 状态数={state_space}, 动作数={action_space}  → Q 表大小 {state_space}×{action_space} = {state_space*action_space} 个数")

    n_training_episodes, learning_rate, gamma = 25000, 0.7, 0.95
    max_steps, max_epsilon, min_epsilon, decay_rate = 99, 1.0, 0.05, 0.005

    # 课程指定的固定评测种子（不要改），保证 100 局从同样的起始布局出发
    eval_seed = [16,54,165,177,191,191,120,80,149,178,48,38,6,125,174,73,50,172,100,148,146,6,25,40,68,148,49,167,9,97,164,176,61,7,54,55,
     161,131,184,51,170,12,120,113,95,126,51,98,36,135,54,82,45,95,89,59,95,124,9,113,58,85,51,134,121,169,105,21,30,11,50,65,12,43,82,145,152,97,106,55,31,85,38,
     112,102,168,123,97,21,83,158,26,80,63,5,81,32,11,28,148]

    Qtable = initialize_q_table(state_space, action_space)
    Qtable = train(n_training_episodes, min_epsilon, max_epsilon, decay_rate, env, max_steps, Qtable)
    mean_r, std_r = evaluate_agent(env, max_steps, 100, Qtable, eval_seed)
    print(f"[Taxi] mean_reward = {mean_r:.2f} +/- {std_r:.2f}  (课程达标线约 >= 4.0，理想 8~9)")


if __name__ == "__main__":
    run_frozenlake()
    run_taxi()
    print("\n完成。注意：全程没有一个神经网络、没有 GPU —— 强化学习最经典的样子。")
