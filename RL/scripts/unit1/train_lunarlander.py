"""
Unit 1 作业：训练第一个深度强化学习 agent —— 让月球着陆器平稳降落。
本文件 = HuggingFace Deep RL Course 的 unit1.ipynb 逐格搬成 Python 脚本，
每一步都加了中文注释解释「这行在干什么」。代码本身和课程 solution 一字不差。

跑法：
    python train_lunarlander.py            # 完整训练 100 万步（M5 约 10 分钟）
    python train_lunarlander.py 20000      # 先小跑 2 万步，验证流程能通（约十几秒）
"""

import sys
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

# 训练步数：默认 100 万（课程要求，评测线 >=200 分）。命令行给个数字就用那个数字先试跑。
TIMESTEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 1_000_000


# ────────────────────────────────────────────────────────────────────
# 第 1 步：认识环境（notebook cell 28 / 30）
# 先不训练，只看看这个环境「长什么样」——RL 一切都从环境的输入输出开始。
# ────────────────────────────────────────────────────────────────────
env = gym.make("LunarLander-v2")
env.reset()
print("_____观测空间 OBSERVATION SPACE_____")
print("Shape:", env.observation_space.shape)          # (8,) —— agent 每一步看到 8 个数
print("随机采样一个观测:", env.observation_space.sample())
# 这 8 个数分别是：着陆器的 x/y 坐标、x/y 速度、机身角度、角速度、左右腿是否触地（2 个布尔）
print("\n_____动作空间 ACTION SPACE_____")
print("动作数量:", env.action_space.n)                # 4 —— 离散动作
print("随机采样一个动作:", env.action_space.sample())
# 4 个动作：0=什么都不做，1=点左引擎，2=点主引擎，3=点右引擎


# ────────────────────────────────────────────────────────────────────
# 第 2 步：创建向量化环境（notebook cell 33）
# 同时并行跑 16 个 LunarLander。为什么？RL 瓶颈在「采样」——agent 要试错几十万步才学得会。
# 16 个环境一起跑，每步就能收集 16 份 (状态,动作,奖励) 经验，采样快 16 倍。
# 这正是上次聊的「RL 贵在采样不在模型」的具体体现。
# ────────────────────────────────────────────────────────────────────
env = make_vec_env("LunarLander-v2", n_envs=16)


# ────────────────────────────────────────────────────────────────────
# 第 3 步：创建 PPO 模型（notebook cell 40，官方 solution 超参数原样照搬）
# ────────────────────────────────────────────────────────────────────
model = PPO(
    policy="MlpPolicy",   # 策略网络用多层感知机（小 MLP）——因为输入是 8 维向量，不是图像
    env=env,
    n_steps=1024,         # 每个环境每次更新前先跑 1024 步收集经验（16 环境 = 16384 步/轮）
    batch_size=64,        # 每个梯度步用 64 条样本
    n_epochs=4,           # 收集来的这批数据反复训练 4 遍
    gamma=0.999,          # 折扣因子：接近 1 = 很看重长远回报（着陆是个长期任务）
    gae_lambda=0.98,      # GAE 参数，用来更稳地估计「优势函数」（这个动作比平均好多少）
    ent_coef=0.01,        # 熵奖励：鼓励 agent 多探索、别过早收敛到一个死板策略
    verbose=1,            # 打印训练过程
    device="cpu",         # ← 关键：这个 MLP 太小，放 CPU 比 GPU/MPS 更快（数据搬去显存的开销 > 计算本身）
)                         #    上次聊的「小模型不需要大算力」，这里就是证据。


# ────────────────────────────────────────────────────────────────────
# 第 4 步：训练并保存（notebook cell 44）
# ────────────────────────────────────────────────────────────────────
print(f"\n开始训练 {TIMESTEPS:,} 步……\n")
model.learn(total_timesteps=TIMESTEPS)

model_name = "ppo-LunarLander-v2"
model.save(model_name)                    # 存成 ppo-LunarLander-v2.zip
print(f"\n模型已保存为 {model_name}.zip")


# ────────────────────────────────────────────────────────────────────
# 第 5 步：评估（notebook cell 48）
# 用一个「干净的、没训练过的」环境跑 10 局，看平均得分。
# Monitor 包一层是为了正确记录每局的总回报。deterministic=True = 评估时不再随机探索，选最优动作。
# 课程通过线：mean_reward >= 200。
# ────────────────────────────────────────────────────────────────────
eval_env = Monitor(gym.make("LunarLander-v2"))
mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10, deterministic=True)
print(f"\n评估结果 mean_reward = {mean_reward:.2f} +/- {std_reward:.2f}")
print("达标(>=200)！" if mean_reward >= 200 else "还没到 200，可以多训练或调参再来一次。")
