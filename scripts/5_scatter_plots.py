# ============================================================
# ข้อ 5) Scatter Plot: Quantity vs Price
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

from pretty import banner, ok, show_kv

banner("STEP 5: Scatter — Quantity vs Price")

df_cleaned = pd.read_csv("data/cleaned_data.csv")
os.makedirs("outputs/figures", exist_ok=True)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sample = df_cleaned.sample(2000, random_state=42)

sns.scatterplot(x="Quantity", y="Price", data=sample, ax=axes[0])
axes[0].set_title("Quantity vs Price (sample 2000 rows)")

sns.scatterplot(x="Quantity", y="Price", data=sample, ax=axes[1])
axes[1].set_xscale("log")
axes[1].set_yscale("log")
axes[1].set_title("Quantity vs Price (Log Scale)")

plt.tight_layout()
plt.savefig("outputs/figures/5_scatter_quantity_price.png", dpi=150)
plt.show()

correlation = df_cleaned["Quantity"].corr(df_cleaned["Price"])
show_kv(
    [
        ("Correlation (Quantity, Price)", f"{correlation:.3f}"),
        ("แปลผล", "ใกล้ 0 = แทบไม่สัมพันธ์โดยตรง"),
        ("กราฟ", "outputs/figures/5_scatter_quantity_price.png"),
    ],
    title="ผลลัพธ์",
)
ok("เสร็จข้อ 5")
