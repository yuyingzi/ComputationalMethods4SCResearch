"""
Unit 8 PI —— PPO 核心练习：手写「裁剪替代目标」(clipped surrogate objective)。
这是 PPO 的灵魂，也是训练 ChatGPT 的 RLHF 里一模一样的那行数学。
不需要 gym、不需要训练，本地秒出结果。

跑：  cd ~/LearningBase/deep-rl-course/unit8 && ../unit1/.venv/bin/python ppo_core_exercise.py
自检全 PASS，就说明你真懂了 clip。然后去 Colab 跑完整 ppo.py 拿证书（见文件末尾）。
"""

import torch


def ppo_policy_loss(new_logprob, old_logprob, advantage, clip_coef=0.2):
    """
    PPO 策略损失（要最小化的 loss = 目标的相反数）。
    参数：
      new_logprob : 当前策略对这些动作的 log 概率（多轮更新中会变）
      old_logprob : 采数据时的旧策略对同样动作的 log 概率（固定）
      advantage   : 优势 A_t，>0 表示这个动作比平均好
      clip_coef   : ε，通常 0.2 —— 允许 ratio 偏离 1 的幅度
    """
  
    # ─────────── TODO：写下面 5 行 ───────────
    # (1) logratio = 新 − 旧的 log 概率
    # (2) ratio = logratio 取指数（= π_new / π_old）
    # (3) pg_loss1 = −advantage × ratio                              （不裁剪）
    # (4) pg_loss2 = −advantage × torch.clamp(ratio, 1-clip_coef, 1+clip_coef)   （裁剪）
    # (5) pg_loss = torch.max(pg_loss1, pg_loss2).mean()
    #     为什么取 max？这是要“最小化”的 loss，取 max 等于对“目标”取 min，
    #     即在“裁剪/不裁剪”两个估计里挑更保守的那个——策略走得太远时不给奖励。
    logratio = new_logprob - old_logprob
    ratio = logratio.exp()   
    pg_loss1 = - advantage * ratio    
    pg_loss2 = - advantage * torch.clamp(ratio, 1-clip_coef, 1+clip_coef)
    pg_loss = torch.max(pg_loss1, pg_loss2).mean()
    return pg_loss


if __name__ == "__main__":
    eps = 0.2

    # 自检①：ratio=1（新旧策略一样）→ 不触发裁剪 → loss = mean(−A·1)
    lp = torch.tensor([0.0, 0.0, 0.0])
    adv = torch.tensor([1.0, 2.0, 3.0])
    out = ppo_policy_loss(lp, lp, adv, eps)
    assert torch.isclose(out, torch.tensor(-2.0)), f"①ratio=1 期望 -2.0，得到 {out.item():.4f}"

    # 自检②：A>0 且 ratio 远超 1+ε → 裁剪生效，用 1+ε=1.2，而不是那个爆掉的大 ratio
    new = torch.tensor([2.0]); old = torch.tensor([0.0]); a = torch.tensor([1.0])  # ratio=e²≈7.39
    out = ppo_policy_loss(new, old, a, eps)
    assert torch.isclose(out, torch.tensor(-1.2)), f"②大ratio 应裁到 -1.2，得到 {out.item():.4f}"

    # 自检③：A<0 且 ratio 远小于 1−ε → 裁剪到 0.8
    new = torch.tensor([-2.0]); old = torch.tensor([0.0]); a = torch.tensor([-1.0])  # ratio≈0.135
    out = ppo_policy_loss(new, old, a, eps)
    assert torch.isclose(out, torch.tensor(0.8)), f"③小ratio 应裁到 0.8，得到 {out.item():.4f}"

    print("✅ 三个自检全 PASS —— 你写对了 PPO 的裁剪目标，这就是 PPO 的心脏。")
    print("   下一步去 Colab 跑完整 ppo.py（unit8_part1.ipynb），拿证书：")
    print('   !python ppo.py --env-id="LunarLander-v2" --repo-id="Yoko999/ppo-CleanRL-LunarLander-v2" --total-timesteps=50000')
