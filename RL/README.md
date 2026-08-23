# Deep Reinforcement Learning Course

Working code and output-free notebooks from the
[Hugging Face Deep RL Course](https://huggingface.co/learn/deep-rl-course/en/unit0/introduction).
Trained models and large run artifacts are published separately on the
[Hugging Face Hub](https://huggingface.co/Yoko999).

## Progress: 9 / 11 — 🎓 certificate of completion earned

The certificate counts 11 hands-on assignments (Units 4, 5 and 8 have two each).
A submission passes when the pushed model's `mean_reward − std_reward` is at or
above the minimum. 9/11 earns the certificate of completion.

| # | Unit | Algorithm / Env | Code | Hub model | Result (mean−std) | Min | Pass |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | Unit 1 | PPO · LunarLander-v2 | [`unit1/`](unit1/) | [ppo-LunarLander-v2](https://huggingface.co/Yoko999/ppo-LunarLander-v2) | 257.94 | 200 | ✅ |
| 2 | Unit 2 | Q-Learning · Taxi-v3 | [`unit2/`](unit2/) | [q-Taxi-v3](https://huggingface.co/Yoko999/q-Taxi-v3) | 4.85 | 4 | ✅ |
| 3 | Unit 3 | DQN · SpaceInvaders | [`unit3/`](unit3/unit3.ipynb) | [dqn-SpaceInvadersNoFrameskip-v4](https://huggingface.co/Yoko999/dqn-SpaceInvadersNoFrameskip-v4) | 405.21 | 200 | ✅ |
| 4 | Unit 4 | REINFORCE · CartPole-v1 | [`unit4/`](unit4/) | [Reinforce-CartPole-v1](https://huggingface.co/Yoko999/Reinforce-CartPole-v1) | 500.00 | 350 | ✅ |
| 5 | Unit 4 | REINFORCE · Pixelcopter | — | — | — | 5 | ☐ |
| 6 | Unit 6 | A2C · PandaReachDense | [`unit6/`](unit6/) | [a2c-PandaReachDense-v3](https://huggingface.co/Yoko999/a2c-PandaReachDense-v3) | −0.29 | −3.5 | ✅ |
| 7 | Unit 8 PI | PPO from scratch · LunarLander | [`unit8/`](unit8/) | [ppo-CleanRL-LunarLander-v2](https://huggingface.co/Yoko999/ppo-CleanRL-LunarLander-v2) | 11.86 | −500 | ✅ |
| 8 | Unit 8 PII | Sample Factory · VizDoom | — | — | — | 5 | ☐ |
| 9 | Unit 5 | ML-Agents · SnowballTarget | [`unit5/`](unit5/unit5.ipynb) | [ppo-SnowballTarget](https://huggingface.co/Yoko999/ppo-SnowballTarget) | 23.45 | −100 | ✅ |
| 10 | Unit 5 | ML-Agents · Pyramids | [`unit5/`](unit5/unit5.ipynb) | [PyramidsRND](https://huggingface.co/Yoko999/PyramidsRND) | 1.64 | −100 | ✅ |
| 11 | Unit 7 | ML-Agents (MA-POCA) · SoccerTwos | [`unit7/`](unit7/) | [poca-SoccerTwos](https://huggingface.co/Yoko999/poca-SoccerTwos) | ELO 1205 | −100 | ✅ |

Extra (beyond the certificate): [a2c-PandaPickAndPlace-v3](https://huggingface.co/Yoko999/a2c-PandaPickAndPlace-v3).

## Notes

- **`unit8/ppo.py`** is a from-scratch CleanRL-style PPO adapted to run on
  **gymnasium + Python 3.12** (the original course code pins the unmaintained
  `gym==0.22`, which no longer builds on modern Python). Its clipped-surrogate
  objective is worked through separately in
  [`unit8/ppo_core_exercise.py`](unit8/ppo_core_exercise.py).
- Units 1, 2, 4, 6, 8-PI and 7 run locally on Apple Silicon. Unit 7's SoccerTwos
  build is a universal (x86_64 + arm64) binary, so it trains natively — see
  [`unit7/`](unit7/) for the local ML-Agents setup. Unit 3 (Atari) and Unit 5
  (SnowballTarget / Pyramids) run on Colab — those Unity binaries are Linux/x86 only.
- Certificate of completion earned at 9/11. For 100% (excellence): Unit 4
  Pixelcopter and Unit 8-PII (VizDoom) remain.
