# ============================================================
# ข้อ 10) แนะนำโปรซื้อคู่ + โมเดลเบาๆ ทำนายสินค้าคู่
# ต้องรันข้อ 8 (และแนะนำข้อ 9) ก่อน
# ============================================================
"""
1) สร้างตารางแนะนำโปรจากคู่สินค้า + เดือนที่ฮิต
2) เทรนโมเดลเบาๆ (Logistic Regression) บนตัวอย่างใบเสร็จ
   เพื่อทำนายว่า "ถ้าตะกร้ามี A แล้วมีโอกาสซื้อ B ไหม"
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pretty import banner, ok, show_df, show_kv, warn

banner("STEP 10: Promo Suggestions + Light Model")

pairs_path = Path("outputs/tables/product_pairs.csv")
best_path = Path("outputs/tables/pairs_best_month.csv")
clean_path = Path("data/cleaned_data.csv")

if not pairs_path.exists():
    warn("รันข้อ 8 ก่อน: python3 scripts/8_market_basket.py")
    raise SystemExit(1)
if not clean_path.exists():
    warn("รันข้อ 4 ก่อน: python3 scripts/4_handle_missing.py")
    raise SystemExit(1)

os.makedirs("outputs/figures", exist_ok=True)
os.makedirs("outputs/tables", exist_ok=True)

pairs = pd.read_csv(pairs_path).head(15)
if best_path.exists():
    best = pd.read_csv(best_path)
    # merge ชื่อคู่
    best["pair_key"] = best["pair"]
else:
    best = None

# ---------- ตารางแนะนำโปร ----------
promo_rows = []
for i, r in pairs.iterrows():
    label = f"{r['name_A'][:28]} + {r['name_B'][:28]}"
    month = "—"
    if best is not None:
        hit = best[best["pair"] == label]
        # label ในข้อ 9 ถูกตัดชื่อเหลือ 22 ตัว — แม็พแบบยืดหยุ่น
        if hit.empty:
            hit = best[best["pair"].str.contains(r["name_A"][:15], regex=False, na=False)]
        if not hit.empty:
            month = str(hit.iloc[0]["best_month"])
    discount = "15%" if r["lift"] >= 3 else ("10%" if r["lift"] >= 2 else "5%")
    promo_rows.append(
        {
            "rank": i + 1,
            "product_A": r["name_A"][:35],
            "product_B": r["name_B"][:35],
            "lift": r["lift"],
            "confidence": r["confidence"],
            "best_month": month,
            "suggested_discount": discount,
            "action": f"โปรซื้อคู่ {discount} ช่วง {month}",
        }
    )

promo = pd.DataFrame(promo_rows)
promo_path = Path("outputs/tables/promo_suggestions.csv")
promo.to_csv(promo_path, index=False)
show_df(promo.drop(columns=["action"]), title="แนะนำโปรซื้อคู่", max_rows=15)

# ---------- โมเดลเบาๆ ----------
ok("กำลังเทรนโมเดลเบาๆ (สุ่มใบเสร็จ + คู่ Top)...")
df = pd.read_csv(clean_path, usecols=["Invoice", "StockCode", "InvoiceDate"])
df["StockCode"] = df["StockCode"].astype(str).str.strip()
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
df["month"] = df["InvoiceDate"].dt.month

# ใช้คู่ Top 1 เป็นตัวอย่างโมเดล (A → B)
top = pairs.iloc[0]
item_a, item_b = str(top["item_A"]), str(top["item_B"])
name_a, name_b = top["name_A"], top["name_B"]

baskets = df.groupby("Invoice").agg(
    items=("StockCode", lambda x: set(x.unique())),
    month=("month", "first"),
)
# เฉพาะใบที่มี A หรือใกล้เคียง (เพื่อไม่ imbalance เกินไป) — ใช้ใบที่มี >=2 สินค้า
baskets = baskets[baskets["items"].map(len) >= 2]

# สุ่มเพื่อเบาเครื่อง
baskets = baskets.sample(n=min(40000, len(baskets)), random_state=42)

# ฟีเจอร์ง่ายๆ: มี A ไหม, เดือน (one-hot แบบตัวเลขเดือน), ขนาดตะกร้า
X_rows = []
y_rows = []
for _, row in baskets.iterrows():
    items = row["items"]
    has_a = 1 if item_a in items else 0
    has_b = 1 if item_b in items else 0
    # สนใจเคสที่มี A เพื่อทำนาย B — และตัวอย่างลบที่ไม่มี A
    X_rows.append([has_a, row["month"], len(items)])
    # เป้า: มี B ในตะกร้าหรือไม่
    y_rows.append(has_b)

X = np.array(X_rows)
y = np.array(y_rows)

# ถ้าคลาสเดียว มีปัญหา — ข้ามโมเดล
if len(np.unique(y)) < 2 or y.sum() < 30:
    warn("ข้อมูลคู่ Top น้อยเกินไปสำหรับโมเดล — ใช้ตารางโปรจาก Rules ได้เลย")
else:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    model = LogisticRegression(max_iter=500)
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, proba)

    # สถานการณ์ตัวอย่าง: มี A ในเดือนต่างๆ
    demo = []
    for m in range(1, 13):
        p = model.predict_proba([[1, m, 5]])[0, 1]
        demo.append({"month": m, "prob_buy_B_if_has_A": round(float(p), 3)})
    demo_df = pd.DataFrame(demo).sort_values("prob_buy_B_if_has_A", ascending=False)

    show_kv(
        [
            ("กฎที่โมเดลโฟกัส", f"{name_a[:30]} → {name_b[:30]}"),
            ("AUC (test)", f"{auc:.3f}"),
            ("แปลผล", "AUC ใกล้ 1 = แยกเคสซื้อคู่ได้ดี"),
        ],
        title="Model performance",
    )
    show_df(demo_df, title="ถ้าตะกร้ามี A → โอกาสซื้อ B ตามเดือน", max_rows=12)

    fig, ax = plt.subplots(figsize=(9, 4), facecolor="white")
    ax.plot(demo_df.sort_values("month")["month"], demo_df.sort_values("month")["prob_buy_B_if_has_A"],
            marker="o", color="#0f766e")
    ax.set_xticks(range(1, 13))
    ax.set_xlabel("Month")
    ax.set_ylabel("P(buy B | has A)")
    ax.set_title(f"Model: if cart has A, chance of buying B\nA={name_a[:28]} | B={name_b[:28]}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig_path = Path("outputs/figures/10_model_pair_by_month.png")
    fig.savefig(fig_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    ok(f"กราฟโมเดล: {fig_path}")

ok(f"ตารางโปร: {promo_path}")
ok("ใช้งาน: เลือกคู่ Lift สูง + เปิดโปรในช่วง best_month")
