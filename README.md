# Credit Default Risk

End-to-end loan default prediction pipeline: feature engineering, baseline and sequence models, calibrated ensembling, SHAP explanations, and a profit-based approval policy.

## Overview
The pipeline (in `finance/`) predicts the probability that an applicant defaults and turns that probability into a lending decision. It is built around Home Credit-style tables (`application_train`, `bureau`, `previous_application`, `installments_payments`, `credit_card_balance`, `POS_CASH_balance`). The raw data is not included in this repo; place it in `finance/data/raw/`.

## Models
- Logistic regression (baseline)
- LightGBM on 124 engineered features
- GRU sequence model on 24-step payment histories
- Calibrated blend of the above (`blend_calibrate.py`)

## Results
Validation metrics from `finance/artifacts/results.json`:

| Model | AUC | Brier |
|---|---|---|
| Logistic | 0.696 | 0.219 |
| LightGBM | 0.708 | 0.107 |
| GRU | 0.714 | 0.217 |
| **Calibrated ensemble** | **0.743** | **0.069** |

The ensemble reaches Gini 0.485 and KS 0.375.

**Business policy** (`finance/reports/policy_summary.txt`): with an assumed 12% profit per good loan and 65% loss given default, the break-even PD is 15.6%. A cut-off of PD <= 0.18 approves 90.8% of applicants and lowers the default rate among approved loans from 7.98% to 6.22%. Model score PSI is 0.0016 (stable).

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
python finance/src/run_all.py
```
Optional: `pyspark` for `features_spark.py`, `imbalanced-learn` for SMOTE.

## Notes
- Fixed seed (42) and 5-fold CV (`config.py`).
- Profit and LGD figures are assumptions, not real portfolio numbers.
