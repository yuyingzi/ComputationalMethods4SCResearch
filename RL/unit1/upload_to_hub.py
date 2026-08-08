"""
Unit 1 最后一步：把训练好的模型上传到 HuggingFace Hub —— 这是拿 certificate 的关键动作。
你贴的那个「Check my progress」Space 就是去 Hub 上查你名下有没有一个 LunarLander-v2 模型、
且平均分 >=200。上传成功后它才能看到你、给你发证。

本文件 = notebook cell 59 的 package_to_hub 照搬 + 逐行中文注释。

【怎么跑】(在你自己的「终端 Terminal」里,两条命令):

  1) 先登录你的 HF 账号(会让你粘贴 write token,token 在 https://huggingface.co/settings/tokens 生成):
     hf auth login

  2) 再运行本脚本,把 YOUR_USERNAME 换成你的 HF 用户名:
     python upload_to_hub.py YOUR_USERNAME
"""

import sys
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv     # 用来把评估环境包成 SB3 要的格式
from huggingface_sb3 import package_to_hub                    # 一键「评估+录视频+生成模型卡+推送」的函数

# ── 读取你的 HF 用户名(命令行第一个参数)。没给就报错提示,避免手滑传错仓库。──
if len(sys.argv) < 2:
    sys.exit("用法：python upload_to_hub.py 你的HF用户名   (例如 python upload_to_hub.py yoko)")
username = sys.argv[1]

# ── 几个基本变量(对应 notebook 里让你填的 TODO)──
env_id = "LunarLander-v2"                              # 环境名字
model_name = "ppo-LunarLander-v2"                      # 本地模型文件名(不带 .zip),要和训练时存的一致
model_architecture = "PPO"                             # 用的算法,会写进模型卡
repo_id = f"{username}/ppo-LunarLander-v2"             # Hub 上的仓库地址 = 用户名/仓库名
commit_message = "Upload PPO LunarLander-v2 trained agent"   # 这次提交的说明

# ── 把训练时存下的 ppo-LunarLander-v2.zip 重新加载回来。──
# (训练是另一个脚本跑的,这里从磁盘读回那个已经练好的模型,不用重训。)
print(f"加载本地模型 {model_name}.zip …")
model = PPO.load(model_name)

# ── 创建「评估用」环境,render_mode='rgb_array' 是为了能把降落过程录成视频帧。──
# DummyVecEnv([lambda: ...]) 是 SB3 要求的写法:哪怕只有 1 个环境也要包成向量化的形式。
eval_env = DummyVecEnv([lambda: gym.make(env_id, render_mode="rgb_array")])

# ── 一键打包上传。这个函数内部会依次做:──
#    ① 用 eval_env 跑几局评估,算出 mean_reward(证书就是看这个分数)
#    ② 录一段降落回放视频 replay.mp4
#    ③ 自动生成 README 模型卡(写上算法、环境、得分)
#    ④ 把模型 + 视频 + 模型卡一起 push 到 https://huggingface.co/{repo_id}
print(f"开始上传到 https://huggingface.co/{repo_id} …(会先评估+录视频,约 1 分钟)")
package_to_hub(
    model=model,                          # 我们训练好的模型
    model_name=model_name,                # 模型名
    model_architecture=model_architecture,# 算法架构:PPO
    env_id=env_id,                        # 环境名
    eval_env=eval_env,                    # 评估环境
    repo_id=repo_id,                      # 目标仓库:你的用户名/ppo-LunarLander-v2
    commit_message=commit_message,        # 提交说明
)
print(f"\n✅ 完成!去 https://huggingface.co/{repo_id} 看你的模型和降落视频。")
print("   然后回到那个 Check-my-progress 页面刷新,就能看到 Unit 1 通过了。")
