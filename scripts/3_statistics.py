# ============================================================
# ข้อ 3) หา Min, Max, Mean, Median, Mode ของ feature ที่เป็นตัวเลข
# ============================================================
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pretty import banner, ok, show_df

banner("STEP 3: Min / Max / Mean / Median / Mode")

df = pd.read_csv("data/online_retail_II.csv", encoding="ISO-8859-1")
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

stats_summary = (
    pd.DataFrame(
        {
            "feature": numeric_cols,
            "Min": [df[c].min() for c in numeric_cols],
            "Max": [df[c].max() for c in numeric_cols],
            "Mean": [df[c].mean() for c in numeric_cols],
            "Median": [df[c].median() for c in numeric_cols],
            "Mode": [df[c].mode().iloc[0] for c in numeric_cols],
        }
    ).round(2)
)
show_df(stats_summary, title="Numeric summary", max_rows=20)

skew = df[numeric_cols].skew().round(2).reset_index()
skew.columns = ["feature", "skewness"]
show_df(skew, title="Skewness", max_rows=20)

ok("เสร็จข้อ 3")
