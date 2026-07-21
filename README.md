# Corporate Bankruptcy Prediction with Machine Learning

This repository is the simplified notebook-based academic version of the project. It prioritizes transparency and sequential execution. The original engineering-oriented version remains in a separate repository.

## Research Question

How effectively can interpretable and tree-based machine-learning models distinguish failed from alive company-year observations for previously unseen companies, and what trade-offs arise between failure detection and false alarms?

## Methodology

The dataset contains American company-year observations. The raw target `status_label` is encoded as `failed`, where `alive = 0` and `failed = 1`. Predictors are the original financial variables `X1` through `X18`; `company_name`, `year`, and `failed` are excluded from the predictor matrix.

The analysis preserves the original research design:

- company-level outer train/test split with `GroupShuffleSplit`, test share `0.20`, random state `42`;
- company-level internal model-training/validation split, validation share `0.20`, random state `42`;
- no company overlap across either split;
- validation PR-AUC from `average_precision_score` as the main selection metric;
- validation rows are not added back before final-test evaluation;
- final-test data are not used for model selection;
- threshold analysis is validation-only and does not change the main final comparison;
- Notebook 06 only generates paper-ready assets from canonical outputs.

## Notebook Workflow

Run the notebooks in numerical order from a fresh kernel.

| Notebook | Purpose | Inputs | Outputs |
|---|---|---|---|
| `01_data_audit_and_preparation.ipynb` | Audit raw data, encode target, create model dataset | `data/raw/american_bankruptcy.csv` | `model_dataset.csv`, feature dictionary, raw validation summaries |
| `02_company_split_and_eda.ipynb` | Create company-level outer split and EDA outputs | `model_dataset.csv` | `train.csv`, `test.csv`, split and EDA tables, EDA figure |
| `03_model_training_and_validation.ipynb` | Train/select seven models on internal validation | `train.csv` | validation metrics, reports, predictions, model specification |
| `04_final_results_and_interpretation.ipynb` | Rebuild selected models and evaluate final test set | train/test data, model specification | final metrics, reports, predictions, coefficients, importance |
| `05_threshold_and_pca_extensions.ipynb` | Run validation-only threshold analysis and PCA extension | validation predictions, train data | threshold tables/figure, PCA tables/figure |
| `06_generate_paper_assets.ipynb` | Generate paper tables and figures from canonical outputs | outputs from notebooks 01-05 | `outputs/paper/tables/`, `outputs/paper/figures/` |

## Models

The seven main models are:

- Majority-class baseline
- Logistic Regression
- L1 Regularized Logistic Regression
- L2 Regularized Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting

The PCA Logistic Regression model is an extension, not a replacement for the main final-test comparison.

## Setup

Python 3.11 is recommended.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
jupyter lab
```

On Windows:

```bash
.venv\Scripts\activate
```

Command-line execution:

```bash
for notebook in notebooks/0*.ipynb; do
    jupyter nbconvert \
        --to notebook \
        --execute \
        --inplace \
        --ExecutePreprocessor.timeout=900 \
        "$notebook"
done
```

## Outputs

Main analytical CSV files are written to `outputs/tables/`. Supporting figures are written to `outputs/figures/`. Paper-ready tables and figures are written to `outputs/paper/`.

The final paper source is in `reports/paper/`. The presentation file is in `reports/presentation/`.
