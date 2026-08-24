# Unit 8 — Proximal Policy Optimization (PPO)

Both parts run **locally on Apple Silicon**.

## Part I — PPO from scratch (LunarLander)

- [`ppo.py`](ppo.py) — a CleanRL-style PPO. The original course code pins the
  unmaintained `gym==0.22`, which no longer builds on Python 3.12, so this is
  ported to **gymnasium** (with `LunarLander-v2 → v3` version resolution). The
  Hub upload / model-card code is kept verbatim so the certification tags match.
- [`ppo_core_exercise.py`](ppo_core_exercise.py) — the clipped-surrogate
  objective (the heart of PPO) as a small fill-in-the-blank with self-checks.

```bash
pip install "gymnasium[box2d]" torch huggingface_hub imageio-ffmpeg
python ppo.py --env-id "LunarLander-v2" --repo-id "<user>/ppo-CleanRL-LunarLander-v2" --total-timesteps 100000
```

Model: [Yoko999/ppo-CleanRL-LunarLander-v2](https://huggingface.co/Yoko999/ppo-CleanRL-LunarLander-v2) · reward ≈ 82 · threshold −500 ✅

## Part II — VizDoom with Sample Factory (doom_health_gathering_supreme)

CLI-driven (Sample Factory), no algorithm file to commit. Runs locally on arm64 —
`vizdoom` ships an Apple Silicon wheel, so **no compilation** is needed.

```bash
uv venv --python 3.10 .venv-vizdoom
uv pip install --python .venv-vizdoom/bin/python vizdoom sample-factory

# Train (APPO, ~4M steps, a few minutes on an M-series CPU; reward climbs 0 → ~30)
python -m sf_examples.vizdoom.train_vizdoom --algo=APPO \
  --env=doom_health_gathering_supreme --experiment=doom_health_gathering \
  --train_dir=./vizdoom_train --train_for_env_steps=4000000 \
  --num_workers=8 --num_envs_per_worker=8 --device=cpu

# Evaluate + push. torch>=2.6 defaults torch.load(weights_only=True), which
# rejects Sample Factory checkpoints — patch it back to False before enjoy:
python - <<'PY'
import torch, runpy, sys
_orig = torch.load
torch.load = lambda *a, **k: _orig(*a, **{**k, "weights_only": False})
sys.argv = ["enjoy_vizdoom", "--algo=APPO", "--env=doom_health_gathering_supreme",
            "--experiment=doom_health_gathering", "--train_dir=./vizdoom_train",
            "--max_num_episodes=10", "--no_render", "--save_video",
            "--push_to_hub", "--hf_repository=<user>/rl_course_vizdoom_health_gathering_supreme",
            "--device=cpu"]
runpy.run_module("sf_examples.vizdoom.enjoy_vizdoom", run_name="__main__")
PY
```

Model: [Yoko999/rl_course_vizdoom_health_gathering_supreme](https://huggingface.co/Yoko999/rl_course_vizdoom_health_gathering_supreme) · reward 12.73 ± 5.03 · threshold 5 ✅
