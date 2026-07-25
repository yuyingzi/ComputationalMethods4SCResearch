# Sarcasm-Aware Sentiment Classification

## Research question

Does adding an explicit predicted-irony signal improve binary sentiment
classification, or does it mainly add another noisy measurement?

## What is implemented

| Artifact | Purpose |
| --- | --- |
| `LogisticRegPredi.py` | TF-IDF plus logistic-regression sentiment baseline |
| `Pretrain.py` | BERT sentiment baseline |
| `EnhancedwithSarcasm_pretrain.py` | Trains an irony detector, prefixes reviews with `[IRONY]` or `[NON_IRONY]`, then trains the sentiment model |
| `test1.ipynb` | Thin notebook entry point for the maintained enhanced script |

The previous homework catalogue was removed because those implementations are
not present in this repository.

## Data

- Sentiment models download the
  [`stanfordnlp/imdb`](https://huggingface.co/datasets/stanfordnlp/imdb)
  train/test dataset. Cite Maas et al., *Learning Word Vectors for Sentiment
  Analysis* (ACL 2011).
- The irony material is based in part on
  [SemEval-2018 Task 3](https://aclanthology.org/S18-1005/). Cite Van Hee,
  Lefever, and Hoste (SemEval 2018).
- `combined_new_irony.csv` contains additional combined records whose exact
  provenance and redistribution terms still need to be documented. Treat it as
  course material only.

## Run

From this directory:

```bash
python LogisticRegPredi.py
python Pretrain.py
python EnhancedwithSarcasm_pretrain.py
```

The BERT scripts download pretrained weights and are compute-intensive; a GPU
is recommended.

## Interpretation boundary

The `[IRONY]` prefix proves that the second model can consume an irony signal.
It does not prove that irony is measured correctly, nor that any performance
change generalizes beyond the datasets used here. Compare the enhanced model
against the BERT baseline before making a substantive claim.
