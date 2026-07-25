# Computational Methods for Social Science

> A reproducibility-first collection of computational methods accumulated
> across projects for use in social science research.

This repository is a growing methods collection rather than a portfolio of
finished projects. Each project is a worked example through which a method is
tested, understood, and documented for possible reuse in future social science
research.

## Repository positioning

### What this repository is

- A collection of computational methods learned and refined through individual
  projects.
- A set of worked examples covering sensor time series, text classification,
  irony detection, and neural models.
- A reference for choosing, adapting, and evaluating methods in later social
  science research.

### What this repository is not

- It is not a portfolio intended mainly to showcase completed work.
- It is not a production library or a unified research framework.
- It is not a benchmark claiming state-of-the-art results.
- It is not the replication package for one paper.
- The reinforcement-learning notebook is external course material, not an
  original research contribution.

## Method index

| Worked example | Possible social science use | Reusable methods | Main artifact | Scope |
| --- | --- | --- | --- | --- |
| Human activity recognition | Turning repeated sensor measurements into behavioral categories | Time-series feature extraction, bootstrap intervals, logistic regression, L1 selection, Naive Bayes | [`AReM-Analysis/MainAnalysis.ipynb`](AReM-Analysis/MainAnalysis.ipynb) | Data bundled; evaluation path validated; rerun for results |
| Movie-review sentiment | Measuring evaluative stance in text | Tokenization, embeddings, MLP, CNN, LSTM | [`final_project/Final_project.ipynb`](final_project/Final_project.ipynb) | Data bundled; split validated; training is compute-intensive |
| Sarcasm-aware sentiment | Testing how contextual language complicates text measurement | TF-IDF logistic baseline, BERT, irony-prefix augmentation | [`EnhancedSentimentwithSarcasm/`](EnhancedSentimentwithSarcasm/) | Experimental; downloads IMDb and pretrained models |
| Reinforcement-learning exercise | Learning sequential decision methods that may support later research designs | PPO, Unity ML-Agents, Hugging Face Hub | [`RL/notebooks/bonus-unit1/`](RL/notebooks/bonus-unit1/) | External tutorial; Colab-oriented; adjacent learning material |

## Research stance

The strongest use of machine learning in computational social science is often
measurement: extracting a variable, label, or pattern that supports a larger
substantive argument. Accordingly:

- held-out performance is evidence about generalization, not causal
  explanation;
- larger or more complex models are not assumed to produce better social
  understanding;
- irony and context-heavy language are treated as measurement challenges, not
  solved constructs;
- saved notebook outputs are cleared so results cannot drift away from the
  current source.

## Quick start

Python 3.10 or 3.11 is recommended.

```bash
git clone https://github.com/yuyingzi/ComputationalMethods4SCResearch.git
cd ComputationalMethods4SCResearch

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python validate_repo.py
jupyter lab
```

The integrity check is fast and does not train models. The BERT experiments
download the IMDb dataset and pretrained weights on first use; a GPU is
recommended.

### Run the NLP scripts

```bash
cd EnhancedSentimentwithSarcasm
python LogisticRegPredi.py
python Pretrain.py
python EnhancedwithSarcasm_pretrain.py
```

## Evaluation contract

- Model and feature selection use training or validation data only.
- Held-out test data is evaluated once and is never passed to `fit()`.
- Random splits, model initialization, and bootstrap estimates use fixed seeds.
- Oversampling is applied only to final case-control fitting; it is not allowed
  to leak duplicated observations across cross-validation folds.
- Notebook results should be regenerated before they are reported or cited.

## Data and attribution

| Data or material | Use in this repository | Source and attribution |
| --- | --- | --- |
| AReM sensor data | Human activity classification | [UCI AReM dataset](https://archive.ics.uci.edu/dataset/366/activity%2Brecognition%2Bsystem%2Bbased%2Bon%2Bmultisensor%2Bdata%2Bfusion), DOI `10.24432/C5SS33`, licensed CC BY 4.0 |
| Movie Review Polarity Dataset v2.0 | Bundled positive/negative review archives in `final_project/Data/` | [Cornell Movie Review Data](https://www.cs.cornell.edu/people/pabo/movie-review-data/); cite Pang and Lee (ACL 2004) |
| Large Movie Review Dataset (IMDb) | Downloaded by the standalone sentiment scripts | [Stanford dataset page](https://ai.stanford.edu/~amaas/data/sentiment/) and [`stanfordnlp/imdb`](https://huggingface.co/datasets/stanfordnlp/imdb); cite Maas et al. (ACL 2011) |
| SemEval-2018 Task 3 | Irony-detection training material | [Task paper](https://aclanthology.org/S18-1005/); cite Van Hee, Lefever, and Hoste (SemEval 2018) |
| Huggy / Deep RL course | Reinforcement-learning tutorial notebook | [Hugging Face Deep RL Course](https://huggingface.co/learn/deep-rl-course/en/unitbonus1/how-huggy-works) |

`EnhancedSentimentwithSarcasm/combined_new_irony.csv` also contains additional
combined records whose exact provenance and redistribution terms are not fully
documented. Treat it as course material only; do not publish it as a new
dataset until that provenance is resolved.

## Known limitations

- Full BERT and TensorFlow runs are not part of the fast integrity check.
- Fresh metrics are intentionally not committed in notebook outputs.
- The projects do not yet share one experiment-tracking or reporting format.
- The repository has no top-level license. Dataset licenses do not automatically
  grant a license for the code; the owner should choose one before inviting
  reuse or contributions.

## Validation

```bash
python validate_repo.py
git diff --check
```

The first command checks Python syntax, notebook integrity, bundled archive
counts, and the most important train/test-separation invariants.
