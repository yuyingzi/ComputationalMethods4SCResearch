# Computational Methods for Social Science Research

Coursework and experiments covering time-series classification, sentiment and
sarcasm classification, and reinforcement learning.

## Setup

Python 3.10 or 3.11 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Large NLP models and the IMDb dataset are downloaded from Hugging Face on first
use.

## Projects

- `AReM-Analysis/MainAnalysis.ipynb`: activity classification using the bundled
  AReM sensor data.
- `final_project/Final_project.ipynb`: sentiment classification using the
  bundled positive and negative review archives.
- `EnhancedSentimentwithSarcasm/`: IMDb sentiment baselines and a BERT
  experiment that injects predicted irony as `[IRONY]` or `[NON_IRONY]` input
  tokens.
- `RL/notebooks/bonus-unit1/`: an external Hugging Face Deep RL course notebook.

Start Jupyter from the repository root:

```bash
python validate_repo.py
jupyter lab
```

Run the standalone NLP baselines from their directory:

```bash
cd EnhancedSentimentwithSarcasm
python LogisticRegPredi.py
python Pretrain.py
python EnhancedwithSarcasm_pretrain.py
```

## Evaluation rules

- Model and feature selection use training/validation data only.
- Held-out test data is evaluated once and is never passed to `fit()`.
- Random splits and bootstrap estimates use fixed seeds.
- Saved notebook outputs are cleared so displayed results cannot drift from the
  current source. Re-run a notebook to produce fresh results.

The repository does not currently declare a license; the owner should choose
one before accepting external contributions or redistribution.
