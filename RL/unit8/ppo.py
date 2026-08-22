# Unit 8 PI — PPO (CleanRL 风格), 适配 gymnasium + Python 3.12。
# 上传函数(package_to_hub / generate_metadata 等)照搬课程原版，证书格式不变。
# 训练/评估改成 gymnasium 新 API；strtobool 内联(distutils 在 3.12 被移除)。

import argparse
import os
import random
import time

import gymnasium as gym          # ← 课程原版是老 gym，这里换成 gymnasium
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical

from huggingface_hub import HfApi, upload_folder
from huggingface_hub.repocard import metadata_eval_result, metadata_save

from pathlib import Path
import datetime
import tempfile
import json
import shutil
import imageio


def strtobool(x):   # 原版 from distutils.util import strtobool；3.12 没了，内联一个
    return str(x).lower() in ("1", "true", "t", "yes", "y", "on")


device = torch.device("cpu")   # __main__ 里会按 --cuda 重设


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--exp-name", type=str, default="ppo")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--env-id", type=str, default="LunarLander-v2")
    p.add_argument("--total-timesteps", type=int, default=50000)
    p.add_argument("--learning-rate", type=float, default=2.5e-4)
    p.add_argument("--num-envs", type=int, default=4)
    p.add_argument("--num-steps", type=int, default=128)
    p.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--num-minibatches", type=int, default=4)
    p.add_argument("--update-epochs", type=int, default=4)
    p.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True)
    p.add_argument("--clip-coef", type=float, default=0.2)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True)
    p.add_argument("--repo-id", type=str, default="ThomasSimonini/ppo-LunarLander-v2")
    args = p.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    return args


# ════════════════════════════════════════════════════════════════════
# 上传相关：以下函数与训练用的 gym 无关，照搬课程原版（证书认这个格式）
# ════════════════════════════════════════════════════════════════════
def package_to_hub(repo_id, model, hyperparameters, eval_env, video_fps=30,
                   commit_message="Push agent to the Hub", token=None, logs=None):
    print("Saving, evaluating, recording video, and pushing to the Hub…")
    repo_url = HfApi().create_repo(repo_id=repo_id, token=token, private=False, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmpdirname:
        tmpdirname = Path(tmpdirname)
        torch.save(model.state_dict(), tmpdirname / "model.pt")
        mean_reward, std_reward = _evaluate_agent(eval_env, 10, model)
        evaluate_data = {
            "env_id": hyperparameters.env_id,
            "mean_reward": mean_reward,
            "std_reward": std_reward,
            "n_evaluation_episodes": 10,
            "eval_datetime": datetime.datetime.now().isoformat(),
        }
        with open(tmpdirname / "results.json", "w") as outfile:
            json.dump(evaluate_data, outfile)
        record_video(eval_env, model, tmpdirname / "replay.mp4", video_fps)
        card, metadata = _generate_model_card("PPO", hyperparameters.env_id, mean_reward, std_reward, hyperparameters)
        _save_model_card(tmpdirname, card, metadata)
        if logs:
            _add_logdir(tmpdirname, Path(logs))
        print(f"Pushing repo {repo_id} to the Hugging Face Hub")
        repo_url = upload_folder(repo_id=repo_id, folder_path=tmpdirname, path_in_repo="",
                                 commit_message=commit_message, token=token)
        print(f"Your model is pushed to the Hub: {repo_url}")
    return repo_url


def _generate_model_card(model_name, env_id, mean_reward, std_reward, hyperparameters):
    metadata = generate_metadata(model_name, env_id, mean_reward, std_reward)
    converted_str = "\n".join(str(vars(hyperparameters)).split(", "))
    model_card = f"""
  # PPO Agent Playing {env_id}

  This is a trained model of a PPO agent playing {env_id}.

  # Hyperparameters
  ```python
  {converted_str}
  ```
  """
    return model_card, metadata


def generate_metadata(model_name, env_id, mean_reward, std_reward):
    metadata = {"tags": [env_id, "ppo", "deep-reinforcement-learning",
                         "reinforcement-learning", "custom-implementation", "deep-rl-course"]}
    eval = metadata_eval_result(
        model_pretty_name=model_name, task_pretty_name="reinforcement-learning",
        task_id="reinforcement-learning", metrics_pretty_name="mean_reward",
        metrics_id="mean_reward", metrics_value=f"{mean_reward:.2f} +/- {std_reward:.2f}",
        dataset_pretty_name=env_id, dataset_id=env_id,
    )
    return {**metadata, **eval}


def _save_model_card(local_path, generated_model_card, metadata):
    readme_path = local_path / "README.md"
    readme = generated_model_card
    if readme_path.exists():
        with readme_path.open("r", encoding="utf8") as f:
            readme = f.read()
    with readme_path.open("w", encoding="utf-8") as f:
        f.write(readme)
    metadata_save(readme_path, metadata)


def _add_logdir(local_path, logdir):
    if logdir.exists() and logdir.is_dir():
        repo_logdir = local_path / "logs"
        if repo_logdir.exists():
            shutil.rmtree(repo_logdir)
        shutil.copytree(logdir, repo_logdir)


# ════════════════════════════════════════════════════════════════════
# 评估 / 录像：改成 gymnasium 新 API（reset 返回 (obs,info)，step 返回 5 元组）
# ════════════════════════════════════════════════════════════════════
def _evaluate_agent(env, n_eval_episodes, policy):
    episode_rewards = []
    for _ in range(n_eval_episodes):
        state, _ = env.reset()
        done = False
        total = 0.0
        while not done:
            state_t = torch.Tensor(state).to(device)
            action, _, _, _ = policy.get_action_and_value(state_t)
            state, reward, terminated, truncated, _ = env.step(action.cpu().numpy())
            done = terminated or truncated
            total += reward
        episode_rewards.append(total)
    return float(np.mean(episode_rewards)), float(np.std(episode_rewards))


def record_video(env, policy, out_directory, fps=30):
    images = []
    state, _ = env.reset()
    images.append(env.render())
    done = False
    while not done:
        state_t = torch.Tensor(state).to(device)
        action, _, _, _ = policy.get_action_and_value(state_t)
        state, reward, terminated, truncated, _ = env.step(action.cpu().numpy())
        done = terminated or truncated
        images.append(env.render())
    imageio.mimsave(out_directory, [np.array(img) for img in images], fps=fps)


# ════════════════════════════════════════════════════════════════════
# 网络 + 环境（Agent/layer_init 与课程一致；make_env 改 gymnasium）
# ════════════════════════════════════════════════════════════════════
def make_env(env_id, seed, idx, run_name):
    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env
    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        obs_dim = int(np.array(envs.single_observation_space.shape).prod())
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, envs.single_action_space.n), std=0.01),
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)


if __name__ == "__main__":
    args = parse_args()
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    # 版本解析：老 gymnasium(0.29) 有 LunarLander-v2 就直接用；新版(≥1.0)废弃了 v2 只剩 v3，
    # 就用 v3 训练/评估（同一环境，版本号升级）。但 args.env_id 始终是 v2 → 元数据标 v2 匹配证书。
    from gymnasium.envs.registration import registry
    make_id = args.env_id if args.env_id in registry else \
        {"LunarLander-v2": "LunarLander-v3"}.get(args.env_id, args.env_id)
    print(f"训练用环境: {make_id}（证书标记: {args.env_id}）")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print("Using device:", device)

    envs = gym.vector.SyncVectorEnv([make_env(make_id, args.seed + i, i, run_name)
                                     for i in range(args.num_envs)])
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "只支持离散动作"

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    global_step = 0
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    num_updates = args.total_timesteps // args.batch_size

    for update in range(1, num_updates + 1):
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            optimizer.param_groups[0]["lr"] = frac * args.learning_rate

        # ── 采样一批 rollout ──
        for step in range(args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs = torch.Tensor(next_obs).to(device)
            next_done = torch.Tensor(next_done).to(device)

            if "episode" in infos:   # 打印结束的那些局的总回报（尽力而为，不同版本容错）
                try:
                    r = infos["episode"]["r"]; mask = infos.get("_episode")
                    for i in range(len(r)):
                        if mask is None or mask[i]:
                            print(f"global_step={global_step}, episodic_return={float(r[i]):.1f}")
                except Exception:
                    pass

        # ── GAE 优势估计 ──
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # ── 展平成一个大 batch ──
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)

        # ── 对同一批数据更新 K 遍（PPO 才敢这么做）──
        b_inds = np.arange(args.batch_size)
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                mb_inds = b_inds[start:start + args.minibatch_size]
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # ★ PPO 裁剪目标 —— 就是你在 ppo_core_exercise.py 里写的那 5 行 ★
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                newvalue = newvalue.view(-1)
                v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()
                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

        print(f"update {update}/{num_updates}  global_step={global_step}")

    envs.close()

    # ── 训练完：评估 + 上传（--repo-id SKIP 则只本地评估，不上传，供测试用）──
    if args.repo_id == "SKIP":
        test_env = gym.make(make_id)
        mr, sr = _evaluate_agent(test_env, 5, agent)
        print(f"[SKIP push] 本地评估 mean={mr:.1f} +/- {sr:.1f}")
    else:
        eval_env = gym.make(make_id, render_mode="rgb_array")
        package_to_hub(repo_id=args.repo_id, model=agent, hyperparameters=args, eval_env=eval_env)
