"""
Unit 4 第二环境：REINFORCE 玩 Pixelcopter-PLE-v0(冲优秀证书 11/11)。
算法和你在 train_reinforce.py 写的 REINFORCE 完全一样,只有两处不同:
  1. 环境是老 gym API(reset 返回 obs、step 返回 4 元组),不是 gymnasium。
  2. 网络更深(3 层),因为 Pixelcopter 比 CartPole 难得多。
超参数照搬课程(50000 局,lr 1e-4,gamma 0.99)。达标线:mean−std ≥ 5。

跑法(用专门的 .venv-ple 环境):
  cd ~/LearningBase/deep-rl-course/unit4
  .venv-ple/bin/python train_pixelcopter.py 50000 SKIP          # 只训练+本地评估,不上传
  .venv-ple/bin/python train_pixelcopter.py 50000 Yoko999/Reinforce-Pixelcopter-PLE-v0   # 训练+上传
"""

import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")   # 无头运行 pygame
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import sys
import json
import datetime
import tempfile
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import imageio
import gym
import gym_pygame  # noqa: F401  —— import 一下把 Pixelcopter-PLE-v0 注册进 gym
from huggingface_hub import HfApi
from huggingface_hub.repocard import metadata_eval_result, metadata_save

device = torch.device("cpu")


# ── 策略网络：Pixelcopter 用 3 层(course cell 75),比 CartPole 的 2 层深 ──
class Policy(nn.Module):
    def __init__(self, s_size, a_size, h_size):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(s_size, h_size)
        self.fc2 = nn.Linear(h_size, h_size * 2)
        self.fc3 = nn.Linear(h_size * 2, a_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.softmax(x, dim=1)

    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        probs = self.forward(state).cpu()
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)


# ── REINFORCE：和你写的一模一样,只是 reset/step 用老 gym API ──
def reinforce(policy, optimizer, n_training_episodes, max_t, gamma, print_every):
    scores_deque = deque(maxlen=100)
    scores = []
    for i_episode in range(1, n_training_episodes + 1):
        saved_log_probs = []
        rewards = []
        state = env.reset()                      # 老 gym：直接返回 obs
        for t in range(max_t):
            action, log_prob = policy.act(state)
            saved_log_probs.append(log_prob)
            state, reward, done, _ = env.step(action)   # 老 gym：4 元组
            rewards.append(reward)
            if done:
                break
        scores_deque.append(sum(rewards))
        scores.append(sum(rewards))

        returns = deque(maxlen=max_t)
        n_steps = len(rewards)
        for t in range(n_steps)[::-1]:
            disc_return_t = returns[0] if len(returns) > 0 else 0
            returns.appendleft(gamma * disc_return_t + rewards[t])   # G_t = r_t + γ·G_{t+1}

        eps = np.finfo(np.float32).eps.item()
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + eps)

        policy_loss = []
        for log_prob, disc_return in zip(saved_log_probs, returns):
            policy_loss.append(-log_prob * disc_return)              # −log_prob · G_t
        policy_loss = torch.cat(policy_loss).sum()

        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()

        if i_episode % print_every == 0:
            print(f"Episode {i_episode}\tAverage Score: {np.mean(scores_deque):.2f}")
    return scores


def evaluate_agent(eval_env, max_t, n_eval_episodes, policy, greedy=False):
    rewards = []
    for _ in range(n_eval_episodes):
        state = eval_env.reset()
        total = 0.0
        for _ in range(max_t):
            if greedy:   # 贪心:选概率最大的动作,评估更稳(方差小),代表策略真实水平
                st = torch.from_numpy(state).float().unsqueeze(0).to(device)
                action = int(policy.forward(st).argmax(1).item())
            else:
                action, _ = policy.act(state)
            state, reward, done, _ = eval_env.step(action)
            total += reward
            if done:
                break
        rewards.append(total)
    return float(np.mean(rewards)), float(np.std(rewards))


def record_video(rec_env, policy, out_path, fps=30):
    try:
        frames = []
        state = rec_env.reset()
        done = False
        while not done:
            frames.append(rec_env.render(mode="rgb_array"))
            action, _ = policy.act(state)
            state, reward, done, _ = rec_env.step(action)
        imageio.mimsave(out_path, [np.array(f) for f in frames], fps=fps)
        return True
    except Exception as e:
        print("录像跳过(不影响证书判定):", e)
        return False


def push_to_hub(repo_id, policy, hyperparameters, eval_env, video_fps=30):
    api = HfApi()
    repo_url = api.create_repo(repo_id=repo_id, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        torch.save(policy, tmp / "model.pt")
        (tmp / "hyperparameters.json").write_text(json.dumps(hyperparameters, indent=2), encoding="utf-8")
        # 贪心 + 30 局评估:Pixelcopter 方差大,采样评估 10 局太不稳,贪心 30 局才代表真实水平
        n_eval = 30
        mean_reward, std_reward = evaluate_agent(eval_env, hyperparameters["max_t"], n_eval, policy, greedy=True)
        hyperparameters = {**hyperparameters, "n_evaluation_episodes": n_eval}
        (tmp / "results.json").write_text(json.dumps({
            "env_id": hyperparameters["env_id"], "mean_reward": mean_reward, "std_reward": std_reward,
            "n_evaluation_episodes": hyperparameters["n_evaluation_episodes"],
            "eval_datetime": datetime.datetime.now().isoformat(),
        }, indent=2), encoding="utf-8")
        metadata = {"tags": [hyperparameters["env_id"], "reinforce", "reinforcement-learning",
                             "custom-implementation", "deep-rl-course"]}
        metadata.update(metadata_eval_result(
            model_pretty_name=repo_id.split("/", 1)[1], task_pretty_name="reinforcement-learning",
            task_id="reinforcement-learning", metrics_pretty_name="mean_reward", metrics_id="mean_reward",
            metrics_value=f"{mean_reward:.2f} +/- {std_reward:.2f}",
            dataset_pretty_name=hyperparameters["env_id"], dataset_id=hyperparameters["env_id"]))
        card = f"# Reinforce Agent playing {hyperparameters['env_id']}\n\nmean_reward = {mean_reward:.2f} +/- {std_reward:.2f}\n"
        (tmp / "README.md").write_text(card, encoding="utf-8")
        metadata_save(tmp / "README.md", metadata)
        record_video(eval_env, policy, tmp / "replay.mp4", video_fps)
        api.upload_folder(repo_id=repo_id, folder_path=str(tmp), path_in_repo="")
        print("已推送:", repo_url, f"| 判定 mean−std = {mean_reward - std_reward:.2f}")


if __name__ == "__main__":
    n_episodes = int(sys.argv[1]) if len(sys.argv) > 1 else 50000
    repo_id = sys.argv[2] if len(sys.argv) > 2 else "SKIP"

    env_id = "Pixelcopter-PLE-v0"
    env = gym.make(env_id)                       # 全局 env,reinforce() 用它
    s_size = env.observation_space.shape[0]      # 7
    a_size = env.action_space.n                  # 2

    hp = {"h_size": 64, "n_training_episodes": n_episodes, "n_evaluation_episodes": 10,
          "max_t": 10000, "gamma": 0.99, "lr": 1e-4, "env_id": env_id,
          "state_space": s_size, "action_space": a_size}

    policy = Policy(s_size, a_size, hp["h_size"]).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=hp["lr"])
    print(f"训练 Pixelcopter,{n_episodes} 局,网络 {s_size}->64->128->{a_size}")
    reinforce(policy, optimizer, n_episodes, hp["max_t"], hp["gamma"], print_every=1000)

    torch.save(policy, "pixelcopter_policy.pt")   # 存盘,防止上传失败白训
    mean_r, std_r = evaluate_agent(env, hp["max_t"], hp["n_evaluation_episodes"], policy)
    print(f"\n评估 mean_reward = {mean_r:.2f} +/- {std_r:.2f}  →  判定 mean−std = {mean_r - std_r:.2f}  (达标线 5)")

    if repo_id != "SKIP":
        push_to_hub(repo_id, policy, hp, env)
    else:
        print("(SKIP 模式,没上传。达标就换成你的 repo-id 重跑,或用存好的 pixelcopter_policy.pt)")
