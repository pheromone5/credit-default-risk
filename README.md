# Credit Default Risk

End-to-end loan default prediction pipeline: feature engineering, baseline and sequence models, calibrated ensembling, SHAP explanations, and a profit-based approval policy.

## Overview
The pipeline (in `finance/`) predicts the probability that an applicant defaults and turns that probability into a lending decision. It uses the schema of the [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk) tables (`application_train`, `bureau`, `previous_application`, `installments_payments`, `credit_card_balance`, `POS_CASH_balance`).

The real dataset (about 2.5 GB, behind a Kaggle login) is not included. `synth_data.py` generates synthetic files with the same names, columns and dtypes, so the whole pipeline runs end to end without it. To use the real data, put the CSVs in `finance/data/raw/` and run with `--skip-synth`.

## Models
- Logistic regression (baseline)
- LightGBM on 124 engineered features
- GRU sequence model on 24-step payment histories
- Calibrated blend of the above (`blend_calibrate.py`)

## What the pipeline produces
- Cross-validated and holdout metrics per model: AUC, Gini, KS, PR-AUC, Brier (`artifacts/results.json`)
- SHAP and LightGBM feature importance, plus per-applicant adverse-action reasons
- Decile table and PSI stability checks
- A profit-optimal approval cut-off, from assumed profit per good loan and loss given default (`reports/policy_summary.txt`)

> **Note:** The artifacts committed here come from a run on **synthetic data**. They show that the pipeline works end to end, but they are not real model performance. The synthetic generator deliberately builds in a signal that favours the sequence model, so the model ranking will differ on real data. Re-run with `--skip-synth` on the real Kaggle data before quoting any results.

## Project structure
```
finance/
  src/          data_prep, features (pandas/Spark), train_baseline, train_lgbm,
                train_seq_model, blend_calibrate, explain_shap, validation,
                business_policy, metrics, run_all
  artifacts/    predictions, metrics, SHAP and feature importance, PSI, decile table
  reports/      figures and policy summary
  requirements.txt
```

## Run it
```bash
pip install -r finance/requirements.txt
python finance/src/run_all.py                 # synthetic data, pandas features
python finance/src/run_all.py --skip-synth    # real Kaggle CSVs in finance/data/raw/
python finance/src/run_all.py --engine spark  # PySpark feature aggregation
```
Optional: `pyspark` for `features_spark.py`, `imbalanced-learn` for SMOTE.

## Notes
- Fixed seed (42) and 5-fold CV (`config.py`).
- Profit and LGD figures are assumptions, not real portfolio numbers.
