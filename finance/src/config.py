"""Central configuration. Every other module imports paths and constants from here."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROC = ROOT / "data" / "processed"
ARTIFACTS = ROOT / "artifacts"
FIGURES = ROOT / "reports" / "figures"
REPORTS = ROOT / "reports"

for _p in (DATA_RAW, DATA_PROC, ARTIFACTS, FIGURES):
    _p.mkdir(parents=True, exist_ok=True)

SEED = 42
N_FOLDS = 5
TARGET = "TARGET"
ID = "SK_ID_CURR"

# ---------------------------------------------------------------- sequences
SEQ_LEN = 24            # trailing months fed to the recurrent encoder
SEQ_FEATURES = [        # one row per (customer, month)
    "dpd",              # days past due that month
    "util",             # balance / credit limit
    "pay_ratio",        # amount paid / amount due
    "late_days",        # mean days between due date and actual payment
    "n_payments",       # number of instalment events that month
    "observed",         # 1 if the customer has any record in that month, else 0
]

# ---------------------------------------------------------------- economics
# Used by business_policy.py. These are assumptions, and the README says so.
PROFIT_PER_GOOD_LOAN = 0.12     # net margin as a fraction of principal
LGD = 0.65                      # loss given default, fraction of principal

# ---------------------------------------------------------------- files
RAW_FILES = {
    "application": "application_train.csv",
    "bureau": "bureau.csv",
    "previous": "previous_application.csv",
    "installments": "installments_payments.csv",
    "credit_card": "credit_card_balance.csv",
    "pos_cash": "POS_CASH_balance.csv",
}
