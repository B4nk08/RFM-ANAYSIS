# ============================================================
# Streamlit App — เลือกสินค้า / เดือน ดูคู่ซื้อด้วยกัน
# รัน:  streamlit run scripts/app_promo.py
# ต้องมี outputs/tables/product_pairs.csv (รันข้อ 8 ก่อน)
# ============================================================
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
PAIRS = ROOT / "outputs" / "tables" / "product_pairs.csv"
PROMO = ROOT / "outputs" / "tables" / "promo_suggestions.csv"
CLEAN = ROOT / "data" / "cleaned_data.csv"

st.set_page_config(page_title="Promo Pair Finder", page_icon="🛒", layout="wide")

st.title("🛒 Product Pair & Promo Finder")
st.caption("เลือกสินค้า → ดูว่ามักซื้อคู่กับอะไร · กรองตามเดือนได้")


@st.cache_data
def load_pairs() -> pd.DataFrame:
    if not PAIRS.exists():
        return pd.DataFrame()
    return pd.read_csv(PAIRS)


@st.cache_data
def load_promo() -> pd.DataFrame:
    if not PROMO.exists():
        return pd.DataFrame()
    return pd.read_csv(PROMO)


@st.cache_data
def load_clean_light() -> pd.DataFrame:
    """โหลดเฉพาะคอลัมน์ที่ต้องใช้สำหรับกราฟรายเดือน"""
    if not CLEAN.exists():
        return pd.DataFrame()
    df = pd.read_csv(CLEAN, usecols=["Invoice", "StockCode", "InvoiceDate"])
    df["StockCode"] = df["StockCode"].astype(str).str.strip()
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["YearMonth"] = df["InvoiceDate"].dt.to_period("M").astype(str)
    return df


pairs = load_pairs()
if pairs.empty:
    st.error("ยังไม่มี `outputs/tables/product_pairs.csv` — รันก่อน: `python3 scripts/8_market_basket.py`")
    st.stop()

# รายชื่อสินค้าทั้งหมดจากคู่
products = sorted(set(pairs["name_A"].tolist() + pairs["name_B"].tolist()))
months_all = []
clean = load_clean_light()
if not clean.empty:
    months_all = sorted(clean["YearMonth"].unique().tolist())

# ----- Sidebar -----
with st.sidebar:
    st.header("ตัวกรอง")
    product = st.selectbox("เลือกสินค้า", options=["(ทั้งหมด)"] + products, index=0)
    month = st.selectbox(
        "เลือกเดือน (Year-Month)",
        options=["(ทุกเดือน)"] + months_all,
        index=0,
        help="ใช้กับกราฟรายเดือนของคู่ที่เลือก",
    )
    min_lift = st.slider("Lift ขั้นต่ำ", min_value=1.0, max_value=20.0, value=2.0, step=0.5)
    top_n = st.slider("แสดงกี่คู่", min_value=5, max_value=50, value=15, step=5)

# ----- กรองตารางคู่ -----
view = pairs[pairs["lift"] >= min_lift].copy()
if product != "(ทั้งหมด)":
    view = view[(view["name_A"] == product) | (view["name_B"] == product)]

view = view.sort_values(["lift", "confidence"], ascending=False).head(top_n)

c1, c2, c3 = st.columns(3)
c1.metric("คู่ที่แสดง", f"{len(view):,}")
c2.metric("Lift สูงสุด", f"{view['lift'].max():.2f}" if len(view) else "—")
c3.metric("สินค้าที่เลือก", product if product != "(ทั้งหมด)" else "ทั้งหมด")

st.subheader("คู่สินค้าที่แนะนำ")
if view.empty:
    st.warning("ไม่พบคู่ตามตัวกรอง — ลองลด Lift หรือเลือกสินค้าอื่น")
else:
    show_cols = ["name_A", "name_B", "pair_count", "support", "confidence", "lift"]
    st.dataframe(view[show_cols], use_container_width=True, hide_index=True)

    # เลือกคู่เพื่อดูรายเดือน
    st.subheader("ดูแยกตามเดือน")
    pair_labels = [
        f"{r.name_A}  ↔  {r.name_B}" for r in view.itertuples()
    ]
    chosen = st.selectbox("เลือกคู่", options=pair_labels)

    if chosen and not clean.empty:
        row = view.iloc[pair_labels.index(chosen)]
        code_a, code_b = str(row["item_A"]), str(row["item_B"])

        # ใบเสร็จที่มีทั้ง A และ B
        inv_a = set(clean.loc[clean["StockCode"] == code_a, "Invoice"])
        inv_b = set(clean.loc[clean["StockCode"] == code_b, "Invoice"])
        both = inv_a & inv_b
        month_map = (
            clean.drop_duplicates("Invoice")
            .set_index("Invoice")["YearMonth"]
            .to_dict()
        )
        ym_counts = pd.Series([month_map[i] for i in both if i in month_map]).value_counts().sort_index()

        if month != "(ทุกเดือน)":
            filtered = ym_counts.loc[[month]] if month in ym_counts.index else pd.Series(dtype=int)
            st.write(f"**เดือน {month}:** {int(filtered.sum()) if len(filtered) else 0} ใบเสร็จที่ซื้อคู่นี้")
            chart_data = ym_counts  # ยังโชว์ทั้งช่วง เพื่อเทียบ
        else:
            chart_data = ym_counts

        st.bar_chart(chart_data, height=280)
        if len(ym_counts):
            best = ym_counts.idxmax()
            st.success(f"เดือนที่คู่นี้ฮิตสุด: **{best}** ({int(ym_counts.max())} ใบเสร็จ) → เหมาะเปิดโปรซื้อคู่ช่วงนี้")
    elif clean.empty:
        st.info("ไม่มี cleaned_data.csv — กราฟรายเดือนจะยังไม่ขึ้น (รันข้อ 4 ก่อน)")

# ----- ตารางโปร -----
promo = load_promo()
if not promo.empty:
    st.subheader("ตารางแนะนำโปร (จากข้อ 10)")
    if product != "(ทั้งหมด)":
        promo_v = promo[(promo["product_A"] == product) | (promo["product_B"] == product)]
    else:
        promo_v = promo
    st.dataframe(promo_v, use_container_width=True, hide_index=True)

st.divider()
st.caption("ข้อมูลจาก Market Basket (ข้อ 8–10) · รัน `streamlit run scripts/app_promo.py` จากโฟลเดอร์โปรเจกต์")
