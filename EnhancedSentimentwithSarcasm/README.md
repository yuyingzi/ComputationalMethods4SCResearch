# Sarcasm-Aware Sentiment Classification

Binary sentiment classification on movie reviews, with an experiment that feeds an
**explicit predicted-irony signal** into the sentiment model. The question it probes:
does telling a BERT classifier "this review is (probably) ironic" improve its
sentiment predictions?

## Approach

Two baselines and one enhanced model, all predicting positive / negative sentiment on
the [IMDB](https://huggingface.co/datasets/stanfordnlp/imdb) dataset.

| Model | Script | What it is |
| --- | --- | --- |
| Baseline A | `LogisticRegPredi.py` | TF-IDF features + logistic regression (classical) |
| Baseline B | `Pretrain.py` | `bert-base-uncased` fine-tuned on IMDB (neural) |
| **Enhanced** | `EnhancedwithSarcasm_pretrain.py` | BERT sentiment, with each review prefixed by a predicted irony tag |

### The enhanced pipeline (one script, three stages)

```
1. Train irony detector   BERT on combined_new_irony.csv (tweet → sarcastic 0/1)  →  iron_model/
2. Label the reviews      run that detector over every IMDB review → [IRONY] / [NON_IRONY]
3. Sentiment + signal     prefix each review with its tag, fine-tune a 2nd BERT, evaluate on IMDB test
```

The comparison that matters is **Enhanced vs Baseline B**: same BERT, the only
difference is the irony prefix.

## Files

| Path | Role |
| --- | --- |
| `LogisticRegPredi.py` | Baseline A — TF-IDF + logistic regression |
| `Pretrain.py` | Baseline B — BERT sentiment |
| `EnhancedwithSarcasm_pretrain.py` | Enhanced 3-stage pipeline (irony detector → label → sentiment) |
| `test1.ipynb` | Notebook entry point — just `%run`s the enhanced script |
| `combined_new_irony.csv` | Irony training data (`tweet`, `sarcastic`), ~5.3k rows — **this is what the code loads** |
| `SemEval2018-T3-train-taskA.txt` | Raw source irony data (SemEval-2018 Task 3); reference only, not read by any script |

## Data

- **Sentiment** — [`stanfordnlp/imdb`](https://huggingface.co/datasets/stanfordnlp/imdb),
  downloaded automatically via 🤗 `datasets`. Cite Maas et al., *Learning Word
  Vectors for Sentiment Analysis* (ACL 2011).
- **Irony** — the detector trains on `combined_new_irony.csv`. It is based in part on
  [SemEval-2018 Task 3](https://aclanthology.org/S18-1005/) (cite Van Hee, Lefever &
  Hoste). Note the domain shift: the detector learns from **tweets** and is then applied
  to long **movie reviews**.

## Setup

```bash
pip install torch transformers datasets scikit-learn nltk pandas numpy
```

## Run

From this directory:

```bash
python LogisticRegPredi.py               # Baseline A — CPU, a few minutes
python Pretrain.py                       # Baseline B — downloads BERT, GPU recommended
python EnhancedwithSarcasm_pretrain.py   # Enhanced   — trains 2 BERT models, GPU recommended
```

Each script prints accuracy / F1 at the end. The BERT scripts download pretrained
weights and are compute-intensive — a GPU is strongly recommended. `test1.ipynb`
runs the enhanced pipeline from a notebook.

## Interpretation caveat

The `[IRONY]` prefix only shows that the sentiment model *can consume* an irony signal —
not that the signal is measured correctly, nor that any change generalizes beyond this
data (recall the tweet → review domain shift above). Always compare the enhanced model
against Baseline B before claiming an improvement.
