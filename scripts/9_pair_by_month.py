# ============================================================
# ข้อ 9) คู่สินค้าฮิตแยกตามเดือน — เพื่อจัดโปรตามฤดูกาล
# ต้องรันข้อ 8 ก่อน (มี outputs/tables/product_pairs.csv)
# ============================================================
from __future__ import annotations

import os
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pretty import banner, ok, show_df, show_kv, warn

TOP_PAIRS = 10  # ใช้ Top คู่จากข้อ 8
banner("STEP 9: Product Pairs by Month")

pairs_path = Path("outputs/tables/product_pairs.csv")
clean_path = Path("data/cleaned_data.csv")
if not pairs_path.exists():
    warn("ยังไม่มี product_pairs.csv — รัน: python3 scripts/8_market_basket.py")
    raise SystemExit(1)
if not clean_path.exists():
    warn("ยังไม่มี cleaned_data.csv — รัน: python3 scripts/4_handle_missing.py")
    raise SystemExit(1)

os.makedirs("outputs/figures", exist_ok=True)
os.makedirs("outputs/tables", exist_ok=True)

pairs = pd.read_csv(pairs_path).head(TOP_PAIRS)
# เก็บคู่แบบ frozenset เพื่อแม็พชื่อ
pair_keys = []
pair_labels = {}
for _, r in pairs.iterrows():
    key = frozenset([str(r["item_A"]), str(r["item_B"])])
    pair_keys.append(key)
    pair_labels[key] = f"{r['name_A'][:22]} + {r['name_B'][:22]}"

needed_items = set()
for k in pair_keys:
    needed_items |= set(k)

print("กำลังโหลดข้อมูล + สร้างเดือน...")
df = pd.read_csv(clean_path, usecols=["Invoice", "StockCode", "InvoiceDate"])
df["StockCode"] = df["StockCode"].astype(str).str.strip()
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
df["YearMonth"] = df["InvoiceDate"].dt.to_period("M").astype(str)

# กรองเฉพาะสินค้าที่เกี่ยวข้องกับ Top pairs (เบาลงมาก)
df = df[df["StockCode"].isin(needed_items)]

# ตะกร้าต่อ Invoice + เดือน
grouped = (
    df.groupby(["Invoice", "YearMonth"])["StockCode"]
    .apply(lambda x: frozenset(x.unique()))
    .reset_index(name="basket")
)

# นับว่าแต่ละคู่โผล่กี่ครั้งในแต่ละเดือน
records = []
for _, row in grouped.iterrows():
    basket = row["basket"]
    ym = row["YearMonth"]
    if len(basket) < 2:
        continue
    for key in pair_keys:
        if key.issubset(basket):
            records.append({"YearMonth": ym, "pair": pair_labels[key]})

month_df = pd.DataFrame(records)
if month_df.empty:
    warn("ไม่พบการซื้อคู่ในข้อมูลเดือน — ตรวจ product_pairs อีกครั้ง")
    raise SystemExit(1)

pivot = (
    month_df.groupby(["YearMonth", "pair"])
    .size()
    .reset_index(name="count")
    .pivot(index="pair", columns="YearMonth", values="count")
    .fillna(0)
    .astype(int)
)
# เรียงเดือน
pivot = pivot.reindex(sorted(pivot.columns), axis=1)

# เดือนที่ฮิตสุดต่อคู่
best_month = pivot.idxmax(axis=1)
best_count = pivot.max(axis=1)
summary = pd.DataFrame(
    {
        "pair": pivot.index,
        "best_month": best_month.values,
        "count_in_best_month": best_count.values,
        "total_pair_orders": pivot.sum(axis=1).values,
    }
).sort_values("total_pair_orders", ascending=False)

show_df(summary, title="เดือนที่คู่สินค้าฮิตสุด", max_rows=TOP_PAIRS)

out_csv = Path("outputs/tables/pairs_by_month.csv")
pivot.to_csv(out_csv)
summary.to_csv("outputs/tables/pairs_best_month.csv", index=False)

# Heatmap
fig, ax = plt.subplots(figsize=(14, 6), facecolor="white")
# ถ้าเดือนเยอะเกินไป แสดงเฉพาะปีล่าสุดที่มีข้อมูลครบ หรือสุ่มช่วง
# จำกัดไม่เกิน 18 คอลัมน์ล่าสุดให้อ่านง่าย
if pivot.shape[1] > 18:
    pivot_plot = pivot.iloc[:, -18:]
else:
    pivot_plot = pivot
sns.heatmap(pivot_plot, cmap="YlGnBu", linewidths=0.3, ax=ax)
ax.set_title("Top Product Pairs by Month (count of invoices)")
ax.set_xlabel("Year-Month")
ax.set_ylabel("")
plt.xticks(rotation=45, ha="right", fontsize=8)
plt.yticks(fontsize=8)
fig.tight_layout()
fig_path = Path("outputs/figures/9_pairs_by_month.png")
fig.savefig(fig_path, dpi=140, bbox_inches="tight")
plt.close(fig)

show_kv(
    [
        ("ตารางรายเดือน", str(out_csv)),
        ("สรุปเดือนฮิต", "outputs/tables/pairs_best_month.csv"),
        ("กราฟ", str(fig_path)),
    ],
    title="Outputs",
)
ok("ใช้ best_month ไปเปิดโปรซื้อคู่ในช่วงนั้นได้เลย")
