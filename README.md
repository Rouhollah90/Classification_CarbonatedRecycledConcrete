# CRC Classification with Weighted KNN

This repository is provided to support the implementation of the KNN classifier
presented in the paper "Classifying Accelerated Carbonation Treatment of
Recycled Concrete: Effects of Process Variables on Properties and CO2 Uptake" by
R. Ayazian and D. K. Panesar (2026).

This repository provides a Python workflow for CRC classification experiments
using weighted K-nearest neighbours (KNN). The code is organized into separate
modules for weighted KNN classification, distribution goodness-of-fit checks,
class labelling, K-means validation, cross-validation, sensitivity analysis, and
noise robustness assessment.

The default script uses synthetic data so the repository can be run without any
private dataset.

## Repository Contents

```text
CRC_Classification/
├── crc_classification.py
├── crc_classification_full.py
├── crc_knn_classification/
│   ├── __init__.py
│   ├── data_cleaning.py
│   ├── examples.py
│   ├── goodness_of_fit.py
│   ├── kmeans_validation.py
│   ├── knn.py
│   ├── labeling.py
│   ├── pipeline.py
│   ├── robustness.py
│   ├── sensitivity.py
│   └── validation.py
├── README.md
└── LICENSE
```

## Module Overview

- `crc_classification.py` - command-line runner for the default demo
- `crc_classification_full.py` - standalone single-file version of the workflow
- `crc_knn_classification/knn.py` - weighted KNN and tunable weighted KNN models
- `crc_knn_classification/goodness_of_fit.py` - distribution goodness-of-fit checks
- `crc_knn_classification/labeling.py` - ACT class labelling and label evaluation
- `crc_knn_classification/kmeans_validation.py` - K-means validation against ACT labels
- `crc_knn_classification/validation.py` - cross-validation, tuning, and medoid metrics
- `crc_knn_classification/sensitivity.py` - Spearman rank sensitivity analysis
- `crc_knn_classification/robustness.py` - noise robustness and reliability assessment
- `crc_knn_classification/pipeline.py` - end-to-end weighted KNN workflow
- `crc_knn_classification/examples.py` - synthetic runnable examples

## Installation

Create and activate a virtual environment, then install the dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows, activate the environment with:

```bash
.venv\Scripts\activate
```

## Quick Start

Run the default synthetic weighted KNN pipeline demo:

```bash
python crc_classification.py
```

The script prints cross-validation metrics, medoids, sensitivity results, and
noise reliability results.

## Using Your Own Data

Prepare your feature matrix `X` as a numeric array with shape
`n_samples x n_features`, and prepare class labels `y` with one label per row.

```python
import numpy as np
from crc_knn_classification import WeightedKNN, run_knn_pipeline

X = np.array([
    [2.1, 3.0, 1.2],
    [5.2, 4.1, 3.4],
    [8.0, 6.8, 6.1],
])

y = np.array(["Class_1", "Class_2", "Class_3"])

model = WeightedKNN(k=1, p=1.5)
model.fit(X, y)
predictions = model.predict(X)

results = run_knn_pipeline(X, y, k=1, p=1.5, n_splits=2, M=5)
```

For real experiments, choose `k`, `n_splits`, and the number of noise
simulations based on the dataset size.

## Main Features

- Percentile trimming and winsorisation for distribution cleaning
- Anderson-Darling goodness-of-fit comparison for several distributions
- ACT class labelling from duration and pressure variables
- K-means validation against ACT classes
- Tunable weighted KNN with uniform, inverse-distance, and Gaussian weighting
- Cross-validation with 90% confidence intervals
- Medoid-based cohesion metric
- Spearman rank sensitivity analysis
- Multiplicative Gaussian noise reliability assessment

## License

This project is distributed under the MIT License.
