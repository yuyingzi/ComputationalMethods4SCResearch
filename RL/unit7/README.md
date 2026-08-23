# Unit 7 — SoccerTwos (ML-Agents, MA-POCA self-play)

Trained **locally on Apple Silicon (M-series)** — no Colab needed. The course's
SoccerTwos build is a universal (x86_64 + arm64) binary, so it runs natively.

Model: [Yoko999/poca-SoccerTwos](https://huggingface.co/Yoko999/poca-SoccerTwos)
· final ELO ≈ 1205 · threshold −100 (submission-based) ✅

There is no algorithm file to commit here — Unit 7 is driven entirely by the
`mlagents-learn` / `mlagents-push-to-hf` CLIs plus a Unity environment binary.
The steps below reproduce the run. Large artifacts (the ML-Agents checkout, the
`.app`, the venv, `results/`) are intentionally not committed.

## 1. Environment (Python 3.10 via uv — conda not required)

```bash
git clone --depth 1 https://github.com/Unity-Technologies/ml-agents
uv venv --python 3.10 mlagents-venv
uv pip install --python mlagents-venv/bin/python -e ./ml-agents/ml-agents-envs -e ./ml-agents/ml-agents
```

The course recommends conda for the grpcio dependency, but a native arm64
Python 3.10 from `uv` installs it cleanly. `git-lfs` must also be installed.

## 2. SoccerTwos environment (macOS)

Download the Mac build (a Google Drive folder from the course) and unlock it:

```bash
uv pip install --python mlagents-venv/bin/python gdown
mkdir -p ml-agents/training-envs-executables && cd ml-agents/training-envs-executables
gdown --folder "https://drive.google.com/drive/folders/1h7YB0qwjoxxghApQdEUQmk95ZwIDxrPG"
cd SoccerTwos && unzip -o SoccerTwos.zip
xattr -cr SoccerTwos.app          # clear the quarantine flag
# SoccerTwos.app/Contents/MacOS/SoccerTwos is a universal binary → runs natively on arm64
```

## 3. Train (self-play)

The default config runs 50M steps (4–8 h, for a competitive agent). The
certificate only needs a submitted model (threshold −100), so `max_steps` was
reduced to 500,000 (~15 min on an M-series CPU):

```bash
# in ml-agents/config/poca/SoccerTwos.yaml: max_steps: 50000000 -> 500000
cd ml-agents
../mlagents-venv/bin/mlagents-learn ./config/poca/SoccerTwos.yaml \
  --env=./training-envs-executables/SoccerTwos/SoccerTwos.app \
  --run-id=SoccerTwos --no-graphics --force
```

## 4. Push to the Hub

```bash
hf auth login   # write token, once
../mlagents-venv/bin/mlagents-push-to-hf \
  --run-id=SoccerTwos --local-dir=./results/SoccerTwos \
  --repo-id=Yoko999/poca-SoccerTwos --commit-message="Push SoccerTwos POCA agent"
```

`mlagents-push-to-hf` tags the repo `ML-Agents-SoccerTwos`, which the course's
certification Space reads to mark the unit complete.
