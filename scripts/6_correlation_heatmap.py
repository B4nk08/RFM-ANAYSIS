# ============================================================
# ข้อ 6) Heatmap ความสัมพันธ์ของทุก Feature
# ต้องรันข้อ 4 ก่อน
# ============================================================
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pretty import banner, ok, show_df

banner("STEP 6: Correlation Heatmap")

df_cleaned = pd.read_csv("data/cleaned_data.csv")
os.makedirs("outputs/figures", exist_ok=True)

df_cleaned["TotalPrice"] = df_cleaned["Quantity"] * df_cleaned["Price"]
corr_cols = ["Quantity", "Price", "TotalPrice", "Customer ID"]
corr_matrix = df_cleaned[corr_cols].corr()

plt.figure(figsize=(7, 6))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", center=0)
plt.title("Correlation Heatmap - Online Retail II")
plt.tight_layout()
plt.savefig("outputs/figures/6_correlation_heatmap.png", dpi=150)
plt.show()

corr_view = corr_matrix.round(3).reset_index().rename(columns={"index": "feature"})
show_df(corr_view, title="Correlation matrix", max_rows=20)

ok("Quantity↔TotalPrice สัมพันธ์สูง · Price↔Quantity มักต่ำ · Customer ID ไม่มีความหมายเชิงตัวเลข")
ok("dataset นี้เหมาะทำ RFM ต่อ (Recency / Frequency / Monetary)")

df_cleaned.to_csv("data/cleaned_data.csv", index=False)
ok("อัปเดต data/cleaned_data.csv (มี TotalPrice) แล้ว")
