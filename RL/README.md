# Deep Reinforcement Learning Course

Working code and output-free notebooks from the
[Hugging Face Deep RL Course](https://huggingface.co/learn/deep-rl-course/en/unit0/introduction).
Trained models and large run artifacts are published separately on the
[Hugging Face Hub](https://huggingface.co/Yoko999).

## Progress: 11 / 11 — 🏆 certificate of excellence earned

The certificate counts 11 hands-on assignments (Units 4, 5 and 8 have two each).
A submission passes when the pushed model's `mean_reward − std_reward` is at or
above the minimum. 9/11 earns completion; 11/11 earns the certificate of excellence.

| # | Unit | Algorithm / Env | Code | Hub model | Result (mean−std) | Min | Pass |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | Unit 1 | PPO · LunarLander-v2 | [`unit1/`](unit1/) | [ppo-LunarLander-v2](https://huggingface.co/Yoko999/ppo-LunarLander-v2) | 257.94 | 200 | ✅ |
| 2 | Unit 2 | Q-Learning · Taxi-v3 | [`unit2/`](unit2/) | [q-Taxi-v3](https://huggingface.co/Yoko999/q-Taxi-v3) | 4.85 | 4 | ✅ |
| 3 | Unit 3 | DQN · SpaceInvaders | [`unit3/`](unit3/unit3.ipynb) | [dqn-SpaceInvadersNoFrameskip-v4](https://huggingface.co/Yoko999/dqn-SpaceInvadersNoFrameskip-v4) | 405.21 | 200 | ✅ |
| 4 | Unit 4 | REINFORCE · CartPole-v1 | [`unit4/`](unit4/) | [Reinforce-CartPole-v1](https://huggingface.co/Yoko999/Reinforce-CartPole-v1) | 500.00 | 350 | ✅ |
| 5 | Unit 4 | REINFORCE · Pixelcopter | [`unit4/`](unit4/) | [Reinforce-Pixelcopter-PLE-v0](https://huggingface.co/Yoko999/Reinforce-Pixelcopter-PLE-v0) | 29.40 | 5 | ✅ |
| 6 | Unit 6 | A2C · PandaReachDense | [`unit6/`](unit6/) | [a2c-PandaReachDense-v3](https://huggingface.co/Yoko999/a2c-PandaReachDense-v3) | −0.29 | −3.5 | ✅ |
| 7 | Unit 8 PI | PPO from scratch · LunarLander | [`unit8/`](unit8/) | [ppo-CleanRL-LunarLander-v2](https://huggingface.co/Yoko999/ppo-CleanRL-LunarLander-v2) | 11.86 | −500 | ✅ |
| 8 | Unit 8 PII | Sample Factory · VizDoom | [`unit8/`](unit8/) | [rl_course_vizdoom_health_gathering_supreme](https://huggingface.co/Yoko999/rl_course_vizdoom_health_gathering_supreme) | 12.73 ± 5.03 | 5 | ✅ |
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
- 9 of 11 assignments run locally on Apple Silicon (M-series). Only Unit 3 (Atari)
  and Unit 5 (SnowballTarget / Pyramids) use Colab — those Unity binaries are
  Linux/x86 only. Unit 7's SoccerTwos build is a universal binary, so it trains
  natively (see [`unit7/`](unit7/)).
- **`unit4/train_pixelcopter.py`** runs REINFORCE on Pixelcopter with legacy
  `gym==0.25` + `numpy==1.26` (numpy 2.0 removed `np.bool8`). Pixelcopter is
  high-variance, so a greedy (argmax) evaluation is used for a stable score.
- **`unit8/`** — see [`unit8/README.md`](unit8/README.md) for both Unit 8 Part I
  (PPO from scratch, gymnasium) and Part II (VizDoom / Sample Factory, local on arm64).
- 🏆 All 11 assignments passed → **certificate of excellence**.

## What next

A post-course study path (theory → from-scratch depth → RL×LLM frontier) with
vetted resources: [`LEARNING_PATH.md`](LEARNING_PATH.md).
