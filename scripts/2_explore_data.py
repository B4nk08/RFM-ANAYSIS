# ============================================================
# ข้อ 2) สำรวจข้อมูล: ครบไหม, missing value, ค่าผิดปกติ, ชนิดข้อมูล
# ============================================================
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pretty import banner, ok, show_df, show_kv, warn

banner("STEP 2: Data Exploration")

df = pd.read_csv("data/online_retail_II.csv", encoding="ISO-8859-1")
os.makedirs("outputs/figures", exist_ok=True)

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
text_cols = df.select_dtypes(include=["object"]).columns.tolist()

show_kv(
    [
        ("rows", f"{df.shape[0]:,}"),
        ("columns", df.shape[1]),
        ("numeric", ", ".join(numeric_cols)),
        ("text", ", ".join(text_cols)),
    ],
    title="Overview",
)

dtypes_tbl = pd.DataFrame({"column": df.columns, "dtype": df.dtypes.astype(str).values})
show_df(dtypes_tbl, title="Column dtypes", max_rows=len(dtypes_tbl))

missing = df.isnull().sum()
miss_tbl = pd.DataFrame(
    {
        "column": missing.index,
        "missing": missing.values,
        "pct_%": (missing / len(df) * 100).round(2),
    }
)
miss_tbl = miss_tbl[miss_tbl["missing"] > 0].sort_values("missing", ascending=False)
show_df(miss_tbl.reset_index(drop=True), title="Missing values", max_rows=20)
warn(f"รวม missing ทั้งตาราง: {int(missing.sum()):,}")

outlier = pd.DataFrame(
    {
        "check": ["Quantity < 0", "Price < 0", "Price == 0", "Customer ID missing"],
        "count": [
            int((df["Quantity"] < 0).sum()),
            int((df["Price"] < 0).sum()),
            int((df["Price"] == 0).sum()),
            int(df["Customer ID"].isnull().sum()),
        ],
    }
)
show_df(outlier, title="Outliers / invalid", max_rows=10)

desc = df.describe().reset_index().rename(columns={"index": "stat"})
show_df(desc, title="describe (numeric)", max_rows=20)

# --- กราฟ ---
miss_plot = miss_tbl.set_index("column")
plt.figure(figsize=(8, 5))
bars = plt.bar(miss_plot.index, miss_plot["pct_%"], color="coral")
plt.ylabel("% Missing")
plt.title("Percentage of Missing Values by Column")
plt.xticks(rotation=45)
for bar in bars:
    h = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, h + 0.5, f"{h:.1f}%", ha="center", fontsize=10)
plt.tight_layout()
plt.savefig("outputs/figures/2_missing_values.png", dpi=150)
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
sns.boxplot(y=df["Quantity"], ax=axes[0])
axes[0].set_title("Quantity - Outliers")
sns.boxplot(y=df["Price"], ax=axes[1])
axes[1].set_title("Price - Outliers")
plt.tight_layout()
plt.savefig("outputs/figures/2_outliers_boxplot.png", dpi=150)
plt.show()

ok("เสร็จข้อ 2 — กราฟอยู่ที่ outputs/figures/")
