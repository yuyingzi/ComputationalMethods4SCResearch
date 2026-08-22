"""
Unit 6 作业：用 A2C 训练一个机械臂把末端移到目标点 (PandaReachDense-v3)。
对应 HF Deep RL Course unit6.ipynb，代码照搬官方 solution，每行加中文讲解。

新东西：
  - 环境是「机器人」(panda-gym，基于 PyBullet 物理引擎)，动作是**连续**的（不是离散选几个），
    观测是个**字典**（当前位置、目标位置分开存），所以策略要用 MultiInputPolicy。
  - 算法 A2C = Advantage Actor-Critic：一个网络同时学「策略(actor)」和「价值(critic)」，
    是 REINFORCE 的升级、PPO 的近亲。
  - VecNormalize：把观测和奖励做标准化，机器人任务里对稳定训练很关键。

运行前先装依赖（在 unit1 的 venv 里）：
    ~/LearningBase/deep-rl-course/unit1/.venv/bin/python -m pip install panda-gym
然后：
    ln -sf ../unit1/.venv .venv        # 复用同一个虚拟环境
    .venv/bin/python train_a2c.py
注：PandaReachible 稠密奖励训得快；100 万步在 M5 上大约十几分钟。达标线 mean_reward >= -3.5。
"""

import gymnasium as gym
import panda_gym  # noqa: F401  —— import 一下就会把 Panda 系列环境注册进 gym

from stable_baselines3 import A2C
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.env_util import make_vec_env


env_id = "PandaReachDense-v3"

# ── 先看看环境长什么样（notebook cell 21）──
env = gym.make(env_id)
print("观测空间(是个字典):", env.observation_space)   # 含 observation / achieved_goal / desired_goal
print("动作空间(连续):", env.action_space)             # Box(-1,1,(3,))：xyz 三个方向的移动量


# ── 训练环境：4 个并行 + 标准化包装（notebook cell 30）──
env = make_vec_env(env_id, n_envs=4)
# norm_obs：标准化观测；norm_reward：标准化奖励；clip_obs=10 把极端值裁掉。稳定训练的关键一步。
env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)


# ── A2C 模型（notebook cell 34）──
# MultiInputPolicy：因为观测是字典，需要能吃多个输入的策略网络（内部各自过 MLP 再拼接）。
model = A2C(policy="MultiInputPolicy", env=env, verbose=1)


# ── 训练 + 保存（notebook cell 36-37）──
model.learn(1_000_000)
model.save("a2c-PandaReachDense-v3")
env.save("vec_normalize.pkl")   # ★必须连标准化的统计量一起存，评估时要用同样的均值方差


# ── 评估（notebook cell 39）──
eval_env = DummyVecEnv([lambda: gym.make("PandaReachDense-v3")])
eval_env = VecNormalize.load("vec_normalize.pkl", eval_env)  # 用训练时的统计量
eval_env.render_mode = "rgb_array"
eval_env.training = False        # 评估时不再更新标准化统计量
eval_env.norm_reward = False     # 评估看真实奖励，不标准化

model = A2C.load("a2c-PandaReachDense-v3")
mean_reward, std_reward = evaluate_policy(model, eval_env)
print(f"\nMean reward = {mean_reward:.2f} +/- {std_reward:.2f}  (达标线 >= -3.5，越接近 0 越好)")
