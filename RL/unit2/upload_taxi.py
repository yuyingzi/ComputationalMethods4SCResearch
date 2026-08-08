"""
把 Unit 2 的 Taxi Q 表上传到 Hugging Face Hub —— 这是「完成 Unit 2 作业」的最后一步。
对应 unit2.ipynb 的 record_video / push_to_hub 格，代码照搬官方，加中文注释。

和 Unit 1 的区别：
  Unit 1 是 SB3 模型，用现成的 package_to_hub 一行搞定。
  Unit 2 是我们自己手写的 Q 表（一个 numpy 数组 + 超参数字典），Hub 上没有现成打包器，
  所以课程自定义了 push_to_hub：把 Q 表 pickle 成 q-learning.pkl、评估、录回放视频、写 model card，再推上去。

运行前：
  1. 装视频依赖（只需一次）：
       python -m pip install imageio imageio-ffmpeg
  2. 登录 HF（只需一次，要 write 权限 token）：
       hf auth login
  3. 把下面的 repo_id 改成你自己的用户名，然后：
       python upload_taxi.py
"""

import pickle
import json
import datetime
import random
from pathlib import Path

import numpy as np
import imageio
import gymnasium as gym
from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.repocard import metadata_eval_result, metadata_save

# 复用训练脚本里的函数，不重复造轮子。
# 注意：train / epsilon_greedy_policy 用到 train_qlearning 模块里的全局变量
#      (env / learning_rate / gamma)，所以下面 __main__ 里要先把它们赋值进去。
import train_qlearning as tq
from train_qlearning import initialize_q_table, train, evaluate_agent


# ════════════════════════════════════════════════════════════════════
# 录一段回放视频（notebook cell 67，官方原样）
# 让训练好的 agent 用纯贪心跑一局，把每一帧存成 mp4。
# ════════════════════════════════════════════════════════════════════
def record_video(env, Qtable, out_directory, fps=1):
    images = []
    terminated = False
    truncated = False
    state, info = env.reset(seed=random.randint(0, 500))
    img = env.render()
    images.append(img)
    while not terminated or truncated:   # 课程原始写法（Taxi 会正常终止，够用）
        action = np.argmax(Qtable[state][:])           # 贪心选动作
        state, reward, terminated, truncated, info = env.step(action)
        img = env.render()
        images.append(img)
    imageio.mimsave(out_directory, [np.array(img) for i, img in enumerate(images)], fps=fps)


# ════════════════════════════════════════════════════════════════════
# 完整上传管线（notebook cell 68，官方原样 + 注释）
# ════════════════════════════════════════════════════════════════════
def push_to_hub(repo_id, model, env, video_fps=1, local_repo_path="hub"):
    _, repo_name = repo_id.split("/")
    eval_env = env
    api = HfApi()

    # Step 1: 在 Hub 上建仓库（已存在则跳过）
    repo_url = api.create_repo(repo_id=repo_id, exist_ok=True)

    # Step 2: 把远端仓库下载到本地临时目录
    repo_local_path = Path(snapshot_download(repo_id=repo_id))

    # Step 3: 存模型。FrozenLake 才有 map_name/slippery，Taxi 没有，这段自动跳过
    if env.spec.kwargs.get("map_name"):
        model["map_name"] = env.spec.kwargs.get("map_name")
        if env.spec.kwargs.get("is_slippery", "") == False:
            model["slippery"] = False
    # ★核心：把整个 model 字典（含 Q 表）pickle 成一个文件
    with open((repo_local_path) / "q-learning.pkl", "wb") as f:
        pickle.dump(model, f)

    # Step 4: 评估，把指标写进 results.json
    mean_reward, std_reward = evaluate_agent(
        eval_env, model["max_steps"], model["n_eval_episodes"], model["qtable"], model["eval_seed"]
    )
    evaluate_data = {
        "env_id": model["env_id"],
        "mean_reward": mean_reward,
        "n_eval_episodes": model["n_eval_episodes"],
        "eval_datetime": datetime.datetime.now().isoformat(),
    }
    with open(repo_local_path / "results.json", "w") as outfile:
        json.dump(evaluate_data, outfile)

    # Step 5: 生成 model card（README + 元数据里的评估指标，证书页面就读这个）
    env_name = model["env_id"]
    if env.spec.kwargs.get("map_name"):
        env_name += "-" + env.spec.kwargs.get("map_name")
    if env.spec.kwargs.get("is_slippery", "") == False:
        env_name += "-" + "no_slippery"

    metadata = {"tags": [env_name, "q-learning", "reinforcement-learning", "custom-implementation"]}
    eval = metadata_eval_result(
        model_pretty_name=repo_name,
        task_pretty_name="reinforcement-learning",
        task_id="reinforcement-learning",
        metrics_pretty_name="mean_reward",
        metrics_id="mean_reward",
        metrics_value=f"{mean_reward:.2f} +/- {std_reward:.2f}",
        dataset_pretty_name=env_name,
        dataset_id=env_name,
    )
    metadata = {**metadata, **eval}

    model_card = f"""
  # **Q-Learning** Agent playing1 **{model["env_id"]}**
  This is a trained model of a **Q-Learning** agent playing **{model["env_id"]}** .

  ## Usage

  ```python
  model = load_from_hub(repo_id="{repo_id}", filename="q-learning.pkl")
  env = gym.make(model["env_id"])
  ```
  """

    readme_path = repo_local_path / "README.md"
    readme = model_card
    if readme_path.exists():
        with readme_path.open("r", encoding="utf8") as f:
            readme = f.read()
    with readme_path.open("w", encoding="utf-8") as f:
        f.write(readme)
    metadata_save(readme_path, metadata)    # 把评估指标写进 README 顶部的 YAML

    # Step 6: 录回放视频
    video_path = repo_local_path / "replay.mp4"
    record_video(env, model["qtable"], video_path, video_fps)

    # Step 7: 把整个文件夹（pkl + json + README + mp4）推上 Hub
    api.upload_folder(repo_id=repo_id, folder_path=repo_local_path, path_in_repo=".")
    print("已推送到 Hub，查看：", repo_url)


if __name__ == "__main__":
    # ── 1) 重新训练 Taxi 拿到 Q 表（确定性环境 + 表格，几秒钟）──
    env_id = "Taxi-v3"
    max_steps, n_training_episodes, n_eval_episodes = 99, 25000, 100
    # 把训练要用的全局变量塞进 train_qlearning 模块（它的 train/epsilon_greedy 靠这些全局量工作）
    tq.env = gym.make(env_id)
    tq.learning_rate, tq.gamma = 0.7, 0.95
    max_epsilon, min_epsilon, decay_rate = 1.0, 0.05, 0.005

    # 课程指定的固定评测种子（不要改，保证判定一致）
    eval_seed = [16,54,165,177,191,191,120,80,149,178,48,38,6,125,174,73,50,172,100,148,146,6,25,40,68,148,49,167,9,97,164,176,61,7,54,55,
     161,131,184,51,170,12,120,113,95,126,51,98,36,135,54,82,45,95,89,59,95,124,9,113,58,85,51,134,121,169,105,21,30,11,50,65,12,43,82,145,152,97,106,55,31,85,38,
     112,102,168,123,97,21,83,158,26,80,63,5,81,32,11,28,148]

    Qtable_taxi = initialize_q_table(500, 6)
    Qtable_taxi = train(n_training_episodes, min_epsilon, max_epsilon, decay_rate, tq.env, max_steps, Qtable_taxi)

    # ── 2) 打包成课程要求的 model 字典 ──
    model = {
        "env_id": env_id,
        "max_steps": max_steps,
        "n_training_episodes": n_training_episodes,
        "n_eval_episodes": n_eval_episodes,
        "eval_seed": eval_seed,
        "learning_rate": tq.learning_rate,
        "gamma": tq.gamma,
        "max_epsilon": max_epsilon,
        "min_epsilon": min_epsilon,
        "decay_rate": decay_rate,
        "qtable": Qtable_taxi,
    }

    # ── 3) 上传（改成你自己的用户名！录视频需要 render_mode="rgb_array"）──
    repo_id = "Yoko999/q-Taxi-v3"   # ⚠️ 改这里
    push_to_hub(repo_id, model, gym.make(env_id, render_mode="rgb_array"))
