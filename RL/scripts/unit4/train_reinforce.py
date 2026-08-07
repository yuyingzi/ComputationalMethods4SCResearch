"""
Unit 4 作业：从零手写策略梯度算法 REINFORCE —— 平衡小车上的杆子 (CartPole)。

REINFORCE 的 4 处算法核心已经完成，保留 TODO 注释用于标出原练习位置。
- reset()/step() 用 gymnasium 新 API：state, _ = env.reset()；
  obs, reward, terminated, truncated, _ = env.step(action)

概念回顾（写之前先想清楚）：
  REINFORCE 直接学策略——网络输入状态、输出各动作的概率。
  跑完一整局后，对每一步：回报 G_t 高的动作就调高它的概率。
  loss = Σ (−log π(a|s) · G_t)，其中 G_t 是从该步往后的折扣累计回报。

跑法：  python train_reinforce.py
"""

from collections import deque
import datetime
import json
import tempfile
from pathlib import Path

import imageio
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import gymnasium as gym
from huggingface_hub import HfApi
from huggingface_hub.repocard import metadata_eval_result, metadata_save

device = torch.device("cpu")   # 网络只有两层，CPU 最快


# ════════════════════════════════════════════════════════════════════
# 策略网络：输入状态 → 输出动作概率分布
# ════════════════════════════════════════════════════════════════════
class Policy(nn.Module):
    def __init__(self, s_size, a_size, h_size):
        super(Policy, self).__init__()
        # TODO(1): 定义两个全连接层
        #   self.fc1 —— 从 s_size 维输入 → h_size 个隐藏单元
        self.fc1=nn.Linear(s_size, h_size)
        #   self.fc2 —— 从 h_size → a_size 个动作
        self.fc2=nn.Linear(h_size, a_size)


    def forward(self, x):
        # TODO(2): 前向传播
        #   x 过 fc1 → ReLU 激活 → 过 fc2 → 最后 softmax(dim=1) 变成概率分布
        hidden = F.relu(self.fc1(x))
        logist = self.fc2(hidden)
        prob = F.softmax(logist, dim=1)
        return prob
        #   提示：F.relu(...) 和 F.softmax(..., dim=1)；记得 return

    def act(self, state):
        """按概率采样一个动作，并返回它的 log 概率（更新时要用）。"""
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        probs = self.forward(state).cpu()
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)


# ════════════════════════════════════════════════════════════════════
# REINFORCE 训练
# ════════════════════════════════════════════════════════════════════
def reinforce(policy, optimizer, n_training_episodes, max_t, gamma, print_every):
    scores_deque = deque(maxlen=100)
    scores = []
    for i_episode in range(1, n_training_episodes + 1):
        saved_log_probs = []
        rewards = []
        state, _ = env.reset()

        # —— 跑完一整局，记录每步的 log 概率和奖励（这段给你了）——
        for t in range(max_t):
            action, log_prob = policy.act(state)
            saved_log_probs.append(log_prob)
            state, reward, terminated, truncated, _ = env.step(action)
            rewards.append(reward)
            if terminated or truncated:
                break
        scores_deque.append(sum(rewards))
        scores.append(sum(rewards))

        # —— 从后往前算每步的折扣回报 G_t（动态规划，O(N)）——
        returns = deque(maxlen=max_t)
        n_steps = len(rewards)
        for t in range(n_steps)[::-1]:
            disc_return_t = returns[0] if len(returns) > 0 else 0
            # TODO(3): 算出第 t 步的折扣回报，appendleft 到 returns 最前面。
            #   公式：G_t = reward[t] + gamma * G_{t+1}
            #   这里 disc_return_t 就是 G_{t+1}（上一轮算好的），rewards[t] 是这一步的奖励。
            returns.appendleft(rewards[t] + gamma * disc_return_t)


        # 回报标准化：减均值除标准差，训练更稳（这段给你了）
        eps = np.finfo(np.float32).eps.item()
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + eps)

        # —— 策略梯度损失 ——
        policy_loss = []
        for log_prob, disc_return in zip(saved_log_probs, returns):
            # TODO(4): 往 policy_loss 里加入这一步的损失项。
            #   策略梯度：loss_t = −log_prob × G_t
            #   直觉：回报 G_t 高的动作，加大它的 log 概率（负号因为优化器是“最小化”）。
            policy_loss.append(-log_prob * disc_return)

        policy_loss = torch.cat(policy_loss).sum()

        # —— 反向传播更新——
        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()

        if i_episode % print_every == 0:
            print(f"Episode {i_episode}\tAverage Score: {np.mean(scores_deque):.2f}")
    return scores


def evaluate_agent(eval_env, max_t, n_eval_episodes, policy):
    """评估：跑 N 局看平均总回报。CartPole 每撑一步 +1，满分 500。（给你了）"""
    rewards = []
    for _ in range(n_eval_episodes):
        state, _ = eval_env.reset()
        total = 0
        for _ in range(max_t):
            action, _ = policy.act(state)
            state, reward, terminated, truncated, _ = eval_env.step(action)
            total += reward
            if terminated or truncated:
                break
        rewards.append(total)
    return np.mean(rewards), np.std(rewards)


def record_video(env, policy, output_path, fps=30):
    """录制一局 RGB 回放，供 Hub 模型卡展示。"""
    frames = []
    state, _ = env.reset()
    terminated = truncated = False
    while not (terminated or truncated):
        frame = env.render()
        if frame is not None:
            frames.append(frame)
        action, _ = policy.act(state)
        state, _, terminated, truncated, _ = env.step(action)
    final_frame = env.render()
    if final_frame is not None:
        frames.append(final_frame)
    imageio.mimsave(output_path, frames, fps=fps)


def json_default(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def push_to_hub(repo_id, policy, hyperparameters, eval_env, video_fps=30):
    """评估、生成模型卡和回放，并上传到 Hugging Face Hub。"""
    api = HfApi()
    repo_url = api.create_repo(repo_id=repo_id, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdirname:
        local_dir = Path(tmpdirname)
        torch.save(policy, local_dir / "model.pt")
        (local_dir / "hyperparameters.json").write_text(
            json.dumps(hyperparameters, indent=2, default=json_default), encoding="utf-8"
        )

        mean_reward, std_reward = evaluate_agent(
            eval_env,
            hyperparameters["max_t"],
            hyperparameters["n_evaluation_episodes"],
            policy,
        )
        (local_dir / "results.json").write_text(
            json.dumps(
                {
                    "env_id": hyperparameters["env_id"],
                    "mean_reward": float(mean_reward),
                    "std_reward": float(std_reward),
                    "n_eval_episodes": int(hyperparameters["n_evaluation_episodes"]),
                    "eval_datetime": datetime.datetime.now().isoformat(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        metadata = {"tags": [
            hyperparameters["env_id"],
            "reinforce",
            "reinforcement-learning",
            "custom-implementation",
            "deep-rl-course",
        ]}
        metadata.update(metadata_eval_result(
            model_pretty_name=repo_id.split("/", 1)[1],
            task_pretty_name="reinforcement-learning",
            task_id="reinforcement-learning",
            metrics_pretty_name="mean_reward",
            metrics_id="mean_reward",
            metrics_value=f"{mean_reward:.2f} +/- {std_reward:.2f}",
            dataset_pretty_name=hyperparameters["env_id"],
            dataset_id=hyperparameters["env_id"],
        ))
        readme_path = local_dir / "README.md"
        readme_path.write_text(
            f"# REINFORCE Agent playing {hyperparameters['env_id']}\n\n"
            "Trained from scratch with PyTorch for the Hugging Face Deep RL Course Unit 4.\n",
            encoding="utf-8",
        )
        metadata_save(readme_path, metadata)
        record_video(eval_env, policy, local_dir / "replay.mp4", video_fps)
        api.upload_folder(repo_id=repo_id, folder_path=local_dir, path_in_repo=".")

    print(f"已上传：{repo_url}")


if __name__ == "__main__":
    env_id = "CartPole-v1"
    env = gym.make(env_id)
    eval_env = gym.make(env_id, render_mode="rgb_array")
    s_size = env.observation_space.shape[0]   # 4：小车位置/速度、杆子角度/角速度
    a_size = env.action_space.n               # 2：向左推 / 向右推

    h_size, n_training_episodes, max_t = 16, 1000, 1000
    gamma, lr = 1.0, 1e-2                      # gamma=1.0：CartPole 是短任务，不折扣
    hyperparameters = {
        "h_size": h_size,
        "n_training_episodes": n_training_episodes,
        "n_evaluation_episodes": 10,
        "max_t": max_t,
        "gamma": gamma,
        "lr": lr,
        "env_id": env_id,
        "state_space": s_size,
        "action_space": a_size,
    }

    print(f"状态维度={s_size}, 动作数={a_size}, 网络={s_size}->{h_size}->{a_size}")
    policy = Policy(s_size, a_size, h_size).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=lr)

    reinforce(policy, optimizer, n_training_episodes, max_t, gamma, print_every=100)

    mean_r, std_r = evaluate_agent(eval_env, max_t, hyperparameters["n_evaluation_episodes"], policy)
    print(f"\n评估结果 mean_reward = {mean_r:.2f} +/- {std_r:.2f}  (满分 500；达标线 >= 350)")
    strict_score = mean_r - std_r
    print(f"严格分 mean_reward - std_reward = {strict_score:.2f}")
    if strict_score >= 350:
        print("达标，开始上传……")
        push_to_hub("Yoko999/Reinforce-CartPole-v1", policy, hyperparameters, eval_env)
    else:
        print("严格分未达到 350，本次不上传；可多跑几次取稳定结果。")
    env.close()
    eval_env.close()
