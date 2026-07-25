# Human Activity Classification from Sensor Time Series

## Research question

Can time-domain summary features from a wearable wireless-sensor network
distinguish human activities, and how sensitive are the results to feature
selection and time-series segmentation?

## What is implemented

- extraction of minimum, maximum, mean, median, standard deviation, and
  quartile features from six sensor signals;
- reproducible bootstrap confidence intervals;
- binary bending/non-bending classification;
- multiclass activity classification with logistic regression and Naive Bayes;
- training-only cross-validation for feature and segment-count selection.

The main artifact is [`MainAnalysis.ipynb`](MainAnalysis.ipynb). The dataset is
bundled under `data/AReM/`.

## Data

The [UCI AReM dataset](https://archive.ics.uci.edu/dataset/366/activity%2Brecognition%2Bsystem%2Bbased%2Bon%2Bmultisensor%2Bdata%2Bfusion)
contains six temporal sensor variables recorded while an actor performed
two bending variants, cycling, lying down, sitting, standing, and walking.

Please cite:

> Palumbo, F., Gallicchio, C., Pucci, R., & Micheli, A. (2016). Activity
> Recognition system based on Multisensor data fusion (AReM). UCI Machine
> Learning Repository. https://doi.org/10.24432/C5SS33

The UCI distribution is licensed under CC BY 4.0.

## Interpretation boundary

This is a classification exercise, not evidence that the extracted features
explain human activity. High accuracy shows that the recorded sensor patterns
separate these activities under this dataset and split; it does not establish
generalization to new people, devices, or social settings.
