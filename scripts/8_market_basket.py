# ============================================================
# ข้อ 8) Market Basket — หาคู่สินค้าที่ซื้อด้วยกันบ่อย
# ใช้ data/cleaned_data.csv (ต้องรันข้อ 4 ก่อน)
# ============================================================
"""
วิธีทำงาน (ไม่หนักเครื่อง):
1) สร้างตะกร้า = สินค้าไม่ซ้ำต่อ Invoice
2) โฟกัสสินค้าขายดี Top N
3) นับคู่ที่โผล่ด้วยกัน → Support / Confidence / Lift
"""
from __future__ import annotations

import os
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pretty import banner, ok, show_df, show_kv, warn

# --- ตั้งค่าให้เบาเครื่อง ---
TOP_ITEMS = 120          # ใช้แค่สินค้าขายดี
MIN_PAIR_COUNT = 50      # คู่ต้องโผล่ร่วมกันอย่างน้อยกี่ใบเสร็จ
TOP_SHOW = 20            # โชว์กี่คู่

banner("STEP 8: Market Basket — Product Pairs")

clean_path = Path("data/cleaned_data.csv")
if not clean_path.exists():
    warn("ยังไม่มี cleaned_data.csv — รัน: python3 scripts/4_handle_missing.py")
    raise SystemExit(1)

os.makedirs("outputs/figures", exist_ok=True)
os.makedirs("outputs/tables", exist_ok=True)

print("กำลังโหลดข้อมูล...")
df = pd.read_csv(clean_path, usecols=["Invoice", "StockCode", "Description"])
df["StockCode"] = df["StockCode"].astype(str).str.strip()
df["Description"] = df["Description"].fillna("Unknown").astype(str).str.strip()

# ชื่อสินค้าแบบสั้น (เอา mode ของ Description ต่อ StockCode)
name_map = (
    df.groupby("StockCode")["Description"]
    .agg(lambda s: s.mode().iloc[0] if len(s.mode()) else s.iloc[0])
    .to_dict()
)

# ตะกร้า: set ของ StockCode ต่อ Invoice
baskets = df.groupby("Invoice")["StockCode"].apply(lambda x: frozenset(x.unique()))
baskets = baskets[baskets.map(len) >= 2]
n_baskets = len(baskets)

show_kv(
    [
        ("ใบเสร็จที่ใช้ (มี >= 2 สินค้า)", f"{n_baskets:,}"),
        ("สินค้าไม่ซ้ำทั้งหมด", f"{df['StockCode'].nunique():,}"),
        ("โฟกัส Top items", TOP_ITEMS),
    ],
    title="Overview",
)

# ความถี่สินค้า
item_count = Counter()
for basket in baskets:
    item_count.update(basket)

top_items = {code for code, _ in item_count.most_common(TOP_ITEMS)}
ok(f"ใช้สินค้าขายดี Top {TOP_ITEMS} ในการหาคู่")

# นับคู่ (เฉพาะสินค้าใน top_items)
pair_count: Counter[tuple[str, str]] = Counter()
for basket in baskets:
    items = sorted(basket & top_items)
    if len(items) < 2:
        continue
    for a, b in combinations(items, 2):
        pair_count[(a, b)] += 1

rows = []
for (a, b), cnt in pair_count.items():
    if cnt < MIN_PAIR_COUNT:
        continue
    support = cnt / n_baskets
    conf_a_b = cnt / item_count[a]
    conf_b_a = cnt / item_count[b]
    # lift ใช้ทิศทางที่ confidence สูงกว่า
    support_a = item_count[a] / n_baskets
    support_b = item_count[b] / n_baskets
    lift = support / (support_a * support_b) if support_a * support_b > 0 else 0
    if conf_a_b >= conf_b_a:
        ante, cons, conf = a, b, conf_a_b
    else:
        ante, cons, conf = b, a, conf_b_a
    rows.append(
        {
            "item_A": ante,
            "name_A": name_map.get(ante, ante)[:40],
            "item_B": cons,
            "name_B": name_map.get(cons, cons)[:40],
            "pair_count": cnt,
            "support": round(support, 4),
            "confidence": round(conf, 4),
            "lift": round(lift, 3),
        }
    )

pairs = pd.DataFrame(rows)
if pairs.empty:
    warn("ไม่พบคู่ที่ผ่านเกณฑ์ — ลองลด MIN_PAIR_COUNT")
    raise SystemExit(1)

pairs = pairs.sort_values(["lift", "confidence", "pair_count"], ascending=False).reset_index(drop=True)
out_csv = Path("outputs/tables/product_pairs.csv")
pairs.to_csv(out_csv, index=False)

show_df(
    pairs[["name_A", "name_B", "pair_count", "support", "confidence", "lift"]],
    title=f"Top {TOP_SHOW} product pairs (เรียงตาม Lift)",
    max_rows=TOP_SHOW,
)

# กราฟ Top 15
top15 = pairs.head(15).copy()
labels = [f"{a[:18]} + {b[:18]}" for a, b in zip(top15["name_A"], top15["name_B"])]
fig, ax = plt.subplots(figsize=(10, 6), facecolor="white")
ax.barh(range(len(top15)), top15["lift"].values, color="#0f766e")
ax.set_yticks(range(len(top15)))
ax.set_yticklabels(labels, fontsize=8)
ax.invert_yaxis()
ax.set_xlabel("Lift")
ax.set_title("Top Product Pairs by Lift (Lift > 1 = bought together more than chance)")
fig.tight_layout()
fig_path = Path("outputs/figures/8_top_product_pairs.png")
fig.savefig(fig_path, dpi=140, bbox_inches="tight")
plt.close(fig)

ok(f"บันทึกตาราง: {out_csv}")
ok(f"บันทึกกราฟ: {fig_path}")
ok("Lift > 1 และ Confidence สูง = เหมาะจัดโปรซื้อคู่")
