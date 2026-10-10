# ============================================================
# หน้าเว็บนำเสนอแบบหน้าเดียว — Online Retail II
# รัน:  streamlit run app.py
# ============================================================
from __future__ import annotations

import io
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))

from common import (  # noqa: E402
    DATASET_INFO,
    FEATURE_INFO,
    FIELD_INFO,
    FORECAST_MONTH,
    FUTURE_WORK,
    HOT_QUANTILE,
    MISSING_DECISIONS,
    MODEL_NAME,
    MODEL_REASON,
    RAW_PATH,
    TEST_START,
    build_product_month,
    clean_data,
    dtype_table,
    fig_boxplot,
    fig_confusion,
    fig_feature_importance,
    fig_heatmap,
    fig_missing,
    fig_scatter,
    hit_rate_at_k,
    impute_comparison,
    load_raw,
    missing_table,
    month_label,
    outlier_table,
    predict_hot,
    promo_for_month,
    train_model,
)

st.set_page_config(
    page_title="Online Retail II — Data Mining",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Prompt:wght@300;400;500;600;700&display=swap');
html, body, p, li, h1, h2, h3, h4, label, button, input, [data-testid="stMarkdownContainer"] {
  font-family: 'Prompt', sans-serif !important;
}
[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"] {display: none;}
.block-container {max-width: 1180px; padding-top: 1.5rem; padding-bottom: 4rem;}

.hero {
  background: radial-gradient(circle at 85% 15%, rgba(255,255,255,.18), transparent 40%),
              linear-gradient(135deg, #4338ca 0%, #7c3aed 55%, #0891b2 100%);
  color: #fff; border-radius: 28px; padding: 48px 52px; margin-bottom: 18px;
  box-shadow: 0 20px 40px -18px rgba(67,56,202,.55);
}
.hero .eyebrow {letter-spacing: .18em; font-size: .78rem; opacity: .85; font-weight: 500;}
.hero h1 {color: #fff; font-size: 2.7rem; font-weight: 700; margin: 6px 0 8px; line-height: 1.15;}
.hero p {font-size: 1.05rem; opacity: .92; margin: 0 0 22px; font-weight: 300;}
.chips {display: flex; flex-wrap: wrap; gap: 10px;}
.chips span {background: rgba(255,255,255,.16); border: 1px solid rgba(255,255,255,.25);
  padding: 7px 16px; border-radius: 999px; font-size: .88rem; backdrop-filter: blur(4px);}

.nav {display: flex; gap: 10px; flex-wrap: wrap; margin: 4px 0 8px;}
.nav a {text-decoration: none !important; color: #334155 !important; background: #fff; border: 1px solid #e2e8f0;
  padding: 8px 18px; border-radius: 999px; font-weight: 500; font-size: .92rem; transition: all .15s;}
.nav a:hover {border-color: #6366f1; color: #4f46e5 !important; box-shadow: 0 4px 12px -6px rgba(79,70,229,.5);}

.section {margin: 64px 0 6px; padding-top: 8px;}
.section .tag {display: inline-block; font-size: .75rem; font-weight: 600; letter-spacing: .16em;
  color: #4f46e5; background: #eef2ff; padding: 5px 12px; border-radius: 8px;}
.section h2 {font-size: 2rem; font-weight: 700; margin: 10px 0 4px; color: #0f172a;}
.section p {color: #64748b; margin: 0;}

.step {display: flex; align-items: center; gap: 12px; margin: 36px 0 10px;}
.step .num {background: linear-gradient(135deg, #4f46e5, #7c3aed); color: #fff; min-width: 34px; height: 34px;
  border-radius: 11px; display: inline-flex; align-items: center; justify-content: center; font-weight: 600;}
.step .title {font-size: 1.25rem; font-weight: 600; color: #0f172a;}

.card {background: #fff; border: 1px solid #e2e8f0; border-radius: 18px; padding: 20px 22px; height: 100%;
  box-shadow: 0 1px 3px rgba(15,23,42,.04);}
.card .ttl {font-weight: 600; margin: 0 0 6px; font-size: 1.02rem; color: #0f172a;}
.card.fw {min-height: 215px;}
.flow {display: flex; align-items: stretch; gap: 8px; flex-wrap: wrap; margin: 6px 0 18px;}
.flow .box {flex: 1; min-width: 150px; background: #fff; border: 1px solid #e2e8f0; border-radius: 16px; padding: 14px 16px;}
.flow .lbl {font-size: .8rem; color: #64748b;}
.flow .num {font-size: 1.5rem; font-weight: 700; color: #4f46e5; line-height: 1.4;}
.flow .sub {font-size: .78rem; color: #94a3b8;}
.flow .arrow {align-self: center; color: #a5b4fc; font-size: 1.4rem; font-weight: 700;}
.card .tech {display: inline-block; font-size: .78rem; color: #7c3aed; background: #f5f3ff;
  padding: 3px 10px; border-radius: 6px; margin-bottom: 8px;}
.card p {margin: 0; color: #475569; font-size: .92rem;}

.kv {display: grid; grid-template-columns: 150px 1fr; gap: 10px 18px; background: #fff;
  border: 1px solid #e2e8f0; border-radius: 18px; padding: 22px 26px;}
.kv .k {color: #64748b; font-size: .9rem;}
.kv .v {color: #0f172a; font-weight: 500;}

.note {color: #475569; font-size: .95rem; line-height: 1.75;}
.note b {color: #0f172a;}
.pill {display: inline-block; padding: 2px 10px; border-radius: 999px; font-size: .78rem; font-weight: 600;}
.pill.train {background: #f1f5f9; color: #475569;}
.pill.test {background: #ecfeff; color: #0e7490;}
.pill.forecast {background: #fdf4ff; color: #a21caf;}

div[data-testid="stMetric"] {background: #fff; border: 1px solid #e2e8f0; border-radius: 16px;
  padding: 14px 18px; box-shadow: 0 1px 3px rgba(15,23,42,.04);}
div[data-testid="stMetricValue"] {font-weight: 600;}
[data-testid="stImage"] img {border-radius: 16px; border: 1px solid #e2e8f0;}
.footer {text-align: center; color: #94a3b8; font-size: .85rem; margin-top: 70px;}
</style>
""",
    unsafe_allow_html=True,
)


# ------------------------------------------------------------
# Data (cache ไว้ เลือก dropdown แล้วไม่ต้องคำนวณใหม่)
# ------------------------------------------------------------
@st.cache_data(show_spinner="กำลังโหลดข้อมูลดิบ ~1 ล้านแถว...")
def get_raw() -> pd.DataFrame:
    return load_raw()


@st.cache_data(show_spinner="กำลัง clean ข้อมูล...")
def get_clean():
    return clean_data(get_raw())


@st.cache_data(show_spinner="กำลังสร้างตาราง สินค้า × เดือน...")
def get_product_month() -> pd.DataFrame:
    return build_product_month(get_clean()[0])


@st.cache_resource(show_spinner=f"กำลังเทรน {MODEL_NAME}...")
def get_model():
    pm = get_product_month()
    metrics, eval_model, final_model, info = train_model(pm)
    return metrics, final_model, info, predict_hot(pm, eval_model, final_model)


def _png(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return buf.getvalue()


@st.cache_data(show_spinner=False)
def chart(name: str) -> bytes:
    raw = get_raw()
    clean = get_clean()[0]
    if name == "missing":
        return _png(fig_missing(raw))
    if name == "boxplot":
        return _png(fig_boxplot(raw))
    if name == "scatter":
        return _png(fig_scatter(clean))
    if name == "heatmap":
        return _png(fig_heatmap(clean)[0])
    _, final_model, info, _ = get_model()
    if name == "confusion":
        return _png(fig_confusion(info["cm"]))
    if name == "importance":
        return _png(fig_feature_importance(final_model))
    raise ValueError(name)


def section(anchor: str, tag: str, title: str, desc: str) -> None:
    st.markdown(
        f'<div class="section" id="{anchor}"><span class="tag">{tag}</span><h2>{title}</h2><p>{desc}</p></div>',
        unsafe_allow_html=True,
    )


def step(num: str, title: str) -> None:
    st.markdown(f'<div class="step"><span class="num">{num}</span><span class="title">{title}</span></div>',
                unsafe_allow_html=True)


def note(html: str) -> None:
    st.markdown(f'<div class="note">{html}</div>', unsafe_allow_html=True)


if not RAW_PATH.exists():
    st.error(f"ไม่พบไฟล์ {RAW_PATH} — วาง online_retail_II.csv ไว้ในโฟลเดอร์ data/")
    st.stop()

raw = get_raw()
clean, steps = get_clean()

# ------------------------------------------------------------
# Hero
# ------------------------------------------------------------
st.markdown(
    f"""
<div class="hero">
  <div class="eyebrow">DATA MINING PROJECT</div>
  <h1>Online Retail II<br>จากข้อมูลการขาย สู่โปรโมชันรายเดือน</h1>
  <p>Preprocessing → Data Visualization → เทรน {MODEL_NAME} → จัดโปรสินค้าขายดีแต่ละเดือน</p>
  <div class="chips">
    <span>{len(raw):,} แถว</span><span>{raw.shape[1]} fields</span>
    <span>clean แล้ว {len(clean):,} แถว</span><span>{MODEL_NAME}</span>
  </div>
</div>
<div class="nav">
  <a href="#preprocessing">Preprocessing</a>
  <a href="#visualization">Data Visualization</a>
  <a href="#model">Model</a>
  <a href="#future">Future Work</a>
</div>
""",
    unsafe_allow_html=True,
)

# ============================================================
# PREPROCESSING
# ============================================================
section("preprocessing", "PART 1", "Preprocessing", "เตรียมข้อมูลให้สะอาดก่อนนำไปวิเคราะห์")

step("1", "เตรียม data — ข้อมูลชื่ออะไร")
c1, c2 = st.columns([1.6, 1])
with c1:
    rows = "".join(f'<div class="k">{k}</div><div class="v">{v}</div>' for k, v in DATASET_INFO)
    st.markdown(f'<div class="kv">{rows}</div>', unsafe_allow_html=True)
with c2:
    st.metric("ขนาดไฟล์", f"{RAW_PATH.stat().st_size / 1024**2:,.1f} MB")
    st.metric("แถว × คอลัมน์", f"{raw.shape[0]:,} × {raw.shape[1]}")

step("2", "แสดงข้อมูล 10 แถว · จำนวนข้อมูล · แต่ละ field เก็บอะไร")
st.dataframe(raw.head(10), width="stretch")
m = st.columns(5)
m[0].metric("แถว", f"{len(raw):,}")
m[1].metric("ใบเสร็จ", f"{raw['Invoice'].nunique():,}")
m[2].metric("สินค้า", f"{raw['StockCode'].nunique():,}")
m[3].metric("ลูกค้า", f"{raw['Customer ID'].nunique():,}")
m[4].metric("ประเทศ", raw["Country"].nunique())
st.dataframe(
    pd.DataFrame(
        {
            "field": list(FIELD_INFO),
            "มีข้อมูล (แถว)": [int(raw[c].notna().sum()) for c in FIELD_INFO],
            "เก็บข้อมูลอะไร": list(FIELD_INFO.values()),
        }
    ),
    hide_index=True,
    width="stretch",
)

step("3", "เช็คชนิดข้อมูล")
st.dataframe(dtype_table(raw), hide_index=True, width="stretch")
note(
    "• <b>InvoiceDate</b> เป็นข้อความ (object) → แปลงเป็น datetime ตอน clean<br>"
    "• <b>Customer ID</b> เป็น float เพราะมีค่าว่าง (NaN) → หลังตัดค่าว่างแปลงเป็น int<br>"
    "• <b>Invoice / StockCode</b> เก็บเป็นข้อความ เพราะมีตัวอักษรปน เช่น C489449, 85123A"
)

step("4", "เช็ค Missing Values และ Outliers")
c1, c2 = st.columns(2)
with c1:
    st.dataframe(
        missing_table(raw),
        hide_index=True,
        width="stretch",
        column_config={
            "missing_%": st.column_config.ProgressColumn("missing %", format="%.2f%%", min_value=0, max_value=100)
        },
    )
with c2:
    st.dataframe(outlier_table(raw).round(2), hide_index=True, width="stretch")
    note(f"• แถวที่ซ้ำกันทั้งแถว <b>{raw.duplicated().sum():,}</b> แถว<br>"
         "• Quantity มีค่าสุดโต่ง ±80,995 (สั่งผิดแล้วยกเลิก)")
c1, c2 = st.columns(2)
c1.image(chart("missing"), width="stretch")
c2.image(chart("boxplot"), width="stretch")

step("5", "จัดการ Missing Value ด้วยวิธีอะไร")
st.dataframe(MISSING_DECISIONS, hide_index=True, width="stretch")
note("<b>ทำไม Customer ID ถึงตัดทิ้ง ไม่เติมค่า?</b> ลองเติมด้วย Min / Max / Mean / Median / Mode แล้ว "
     f"ทุกวิธีทำให้ {int(raw['Customer ID'].isna().sum()):,} แถวกลายเป็น <b>ลูกค้าคนเดียว</b> "
     "ยอดซื้อของลูกค้าจึงผิดหมด")
st.dataframe(impute_comparison(raw), hide_index=True, width="stretch")
note("• แถวที่ซ้ำกันทั้งแถว เก็บไว้ เพราะอาจเป็นการกดสั่งสินค้าเดียวกันซ้ำในใบเสร็จเดียว (ยอดขายจริง)")

step("6", "สรุปข้อมูลที่เหลือหลัง Clean")
c1, c2 = st.columns([1, 1.15])
with c1:
    m1, m2 = st.columns(2)
    m1.metric("ก่อน clean", f"{len(raw):,}")
    m2.metric("หลัง clean", f"{len(clean):,}", f"-{len(raw) - len(clean):,} แถว", delta_color="inverse")
    m1.metric("เหลือ", f"{len(clean) / len(raw) * 100:.1f}%")
    m2.metric("Missing ที่เหลือ", int(clean.isna().sum().sum()))
with c2:
    st.dataframe(
        steps,
        hide_index=True,
        width="stretch",
        column_config={
            "แถวคงเหลือ": st.column_config.ProgressColumn(
                "แถวคงเหลือ", format="%d", min_value=0, max_value=int(steps["แถวคงเหลือ"].max())
            )
        },
    )
st.dataframe(clean.head(10), width="stretch")
note("เพิ่มคอลัมน์ <b>TotalPrice = Quantity × Price</b> สำหรับวัดยอดขาย")

# ============================================================
# DATA VISUALIZATION
# ============================================================
section("visualization", "PART 2", "Data Visualization", "ดูความสัมพันธ์ของตัวแปรหลังข้อมูลสะอาดแล้ว")

step("1", "Scatter plot ของ Quantity กับ Price (สเกลปกติ และ log)")
st.image(chart("scatter"), width="stretch")
c1, c2, c3 = st.columns(3)
c1.metric("Correlation", f"{clean['Quantity'].corr(clean['Price']):.3f}")
c2.metric("Median Quantity", f"{clean['Quantity'].median():.0f} ชิ้น")
c3.metric("Median Price", f"£{clean['Price'].median():.2f}")
note("• <b>สเกลปกติ</b> จุดกองที่มุมซ้ายล่าง เพราะมีค่าสุดโต่งไม่กี่ค่าดึงแกนออกไป<br>"
     "• <b>สเกล log</b> เห็นแนวโน้มชัดขึ้น: สินค้าราคาสูงมักถูกซื้อจำนวนน้อย สินค้าราคาถูกซื้อทีละมาก")

step("2", "Correlation heatmap ของ Quantity, Price และ TotalPrice")
fig_corr = fig_heatmap(clean)[1]
plt.close("all")
c1, c2 = st.columns([1.1, 1])
c1.image(chart("heatmap"), width="stretch")
with c2:
    st.dataframe(fig_corr.round(3), width="stretch")
    note(f"• <b>Quantity ↔ TotalPrice = {fig_corr.loc['Quantity', 'TotalPrice']:.2f}</b> สูงสุด "
         "→ จำนวนชิ้นคือตัวขับยอดขาย<br>"
         f"• <b>Price ↔ TotalPrice = {fig_corr.loc['Price', 'TotalPrice']:.2f}</b> ราคาส่งผลต่อยอดขายน้อยกว่า<br>"
         f"• <b>Quantity ↔ Price = {fig_corr.loc['Quantity', 'Price']:.2f}</b> แทบไม่สัมพันธ์เชิงเส้น")

# ============================================================
# MODEL
# ============================================================
section("model", "PART 3", "Model", f"ใช้ {MODEL_NAME} ทำนายว่าสินค้าไหนจะขายดีในแต่ละเดือน")
metrics, final_model, info, pred = get_model()

step("1", f"เทรนข้อมูลด้วย {MODEL_NAME}")
note(MODEL_REASON)
st.write("")

st.markdown("**ข้อมูลที่ใช้เทรน — ใช้ข้อมูลทั้งหมด ไม่สุ่มตัด**")
flow = [
    ("ข้อมูลหลัง clean", f"{len(clean):,}", "1 แถว = สินค้า 1 รายการในใบเสร็จ"),
    ("รวมเป็น สินค้า × เดือน", f"{info['n_rows']:,}", "สรุปยอดขายรายสินค้า รายเดือน"),
    ("Train", f"{info['n_train']:,}", info["train_months"]),
    ("Test", f"{info['n_test']:,}", info["test_months"]),
    ("โมเดลสุดท้าย (Train+Test)", f"{info['n_labeled']:,}", "ใช้พยากรณ์ ธ.ค. 2011"),
]
st.markdown(
    '<div class="flow">'
    + '<span class="arrow">→</span>'.join(
        f'<div class="box"><div class="lbl">{a}</div><div class="num">{b}</div><div class="sub">{c}</div></div>'
        for a, b, c in flow
    )
    + "</div>",
    unsafe_allow_html=True,
)

c1, c2 = st.columns([1, 1])
with c1:
    note(
        "<b>โจทย์:</b> สินค้านี้จะ <b>ขายดี</b> ในเดือนนี้ไหม (Classification 2 คลาส)<br>"
        f"<b>Target (y):</b> is_hot = 1 ถ้ายอดขายเดือนนั้นอยู่ <b>Top {round((1 - HOT_QUANTILE) * 100)}%</b> "
        f"ของสินค้าทั้งหมด (มี {info['hot_rate'] * 100:.1f}% ของข้อมูล)<br>"
        f"<b>ทำไมไม่เทรนตรงๆ บน {len(clean):,} แถว:</b> โจทย์ถามระดับ 'สินค้าในแต่ละเดือน' "
        "จึงต้องรวมยอดทุกใบเสร็จของสินค้านั้นในเดือนนั้นก่อน — ทุกแถวถูกใช้ครบ<br>"
        "<b>แบ่ง Train / Test ตามเวลา:</b> เรียนจากอดีต แล้วทดสอบกับเดือนหลังจากนั้นที่โมเดลไม่เคยเห็น<br>"
        "<b>วัดผล:</b> Accuracy, Precision, Recall, F1, ROC-AUC"
    )
with c2:
    st.dataframe(
        pd.DataFrame({"feature (X)": list(FEATURE_INFO), "ความหมาย": list(FEATURE_INFO.values())}),
        hide_index=True,
        width="stretch",
    )

st.markdown("**ผลบน Test set**")
m = st.columns(5)
for col, k in zip(m, ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC"]):
    col.metric(k.replace("_", "-"), f"{metrics[k]:.3f}")
c1, c2 = st.columns([1, 1.1])
c1.image(chart("confusion"), width="stretch")
with c2:
    cm = info["cm"]
    note(
        f"• ทายว่า <b>ขายดี</b> แล้วถูก <b>{cm[1, 1]:,}</b> สินค้า · ทายว่าขายดีแต่ผิด <b>{cm[0, 1]:,}</b><br>"
        f"• สินค้าขายดีจริงที่ทายพลาด <b>{cm[1, 0]:,}</b> · ทายว่าไม่ขายดีแล้วถูก <b>{cm[0, 0]:,}</b><br><br>"
        f"<b>Precision {metrics['Precision']:.2f}</b> = ที่ทายว่าขายดี ถูกจริง {metrics['Precision'] * 100:.0f}%<br>"
        f"<b>Recall {metrics['Recall']:.2f}</b> = สินค้าขายดีจริง จับได้ {metrics['Recall'] * 100:.0f}%<br>"
        f"<b>ROC-AUC {metrics['ROC_AUC']:.2f}</b> = แยกสินค้าขายดี / ไม่ขายดีได้ดีมาก (1 = สมบูรณ์, 0.5 = เดาสุ่ม)<br>"
        f"Accuracy สูงถึง {metrics['Accuracy']:.2f} ส่วนหนึ่งเพราะสินค้าขายดีมีแค่ ~8% จึงดู F1 และ ROC-AUC ประกอบ"
    )

step("2", "จากผลทำนาย → จัดโปรลดราคาสินค้าขายดีแต่ละเดือน")
test_months = sorted(pred.loc[pred["period"] == "Test", "YearMonth"].unique())
hit = hit_rate_at_k(pred, test_months)
c1, c2 = st.columns([1.1, 1])
img = chart("importance")
c1.image(img, width="stretch")
with c2:
    note(
        "<b>ขั้นตอน</b><br>"
        f"1. ใช้ {MODEL_NAME} ทำนาย <b>โอกาสขายดี</b> ของสินค้าทุกตัวในเดือนที่เลือก<br>"
        "2. เรียงจากโอกาสสูงสุด แล้วเลือก <b>Top 10</b><br>"
        "3. ให้ส่วนลดตามอันดับ: <b>1–3 = 15%</b> · <b>4–6 = 10%</b> · <b>7–10 = 5%</b><br>"
        "4. จัดโปรต้นเดือน เพื่อดึงลูกค้าด้วยสินค้าที่คนต้องการอยู่แล้ว"
    )
    st.metric("Top 10 ที่แนะนำ ขายดีจริงกี่ % (เดือน Test)", f"{hit * 100:.0f}%")
    note("feature ที่สำคัญที่สุดคือ <b>ยอดขายเดือนก่อน / จำนวนลูกค้าเดือนก่อน</b> "
         "→ สินค้าที่ขายดีต่อเนื่องมีแนวโน้มขายดีต่อ")

# ============================================================
# FUTURE WORK
# ============================================================
section("future", "PART 4", "Future Work", "นำผลลัพธ์ไปใช้งานจริง และแนวทางต่อยอด")

step("1", "เลือกเดือน → ดูว่าเดือนนี้ควรจัดโปรสินค้าอะไร")


@st.fragment
def promo_picker() -> None:
    months = sorted(pred["YearMonth"].unique(), reverse=True)
    period_of = pred.groupby("YearMonth")["period"].first().to_dict()
    tag = {"Train": "ข้อมูล Train", "Test": "ข้อมูล Test", "Forecast": "พยากรณ์ล่วงหน้า"}
    c1, c2 = st.columns([1.3, 2])
    ym = c1.selectbox(
        "เลือกเดือน",
        months,
        format_func=lambda x: f"{month_label(x)}  ·  {tag[period_of[x]]}",
    )
    period = period_of[ym]
    with c2:
        desc = {
            "Train": "เดือนที่โมเดลใช้เรียน — ใช้ดูตัวอย่างย้อนหลัง",
            "Test": "เดือนที่โมเดลไม่เคยเห็น — เทียบกับยอดขายจริงได้",
            "Forecast": "เดือนล่าสุด ข้อมูลจริงยังไม่ครบ — คือคำแนะนำล่วงหน้า",
        }[period]
        st.markdown(
            f'<div style="margin-top:30px"><span class="pill {period.lower()}">{period}</span> '
            f'<span class="note">{desc}</span></div>',
            unsafe_allow_html=True,
        )

    table = promo_for_month(pred, ym)
    m = st.columns(3)
    m[0].metric("เดือน", month_label(ym))
    m[1].metric("สินค้าที่ประเมิน", f"{(pred['YearMonth'] == ym).sum():,}")
    if "ขายดีจริง?" in table:
        m[2].metric("แนะนำแล้วขายดีจริง", f"{(table['ขายดีจริง?'] == '✔').sum()} / {len(table)}")
    else:
        m[2].metric("ส่วนลดสูงสุด", "15%")

    st.dataframe(
        table,
        hide_index=True,
        width="stretch",
        column_config={
            "โอกาสขายดี (%)": st.column_config.ProgressColumn(
                "โอกาสขายดี", format="%.1f%%", min_value=0, max_value=100
            ),
            "สินค้า": st.column_config.TextColumn("สินค้า", width="large"),
        },
    )
    st.download_button(
        "ดาวน์โหลดตารางโปรเดือนนี้ (CSV)",
        table.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"promo_{ym}.csv",
        mime="text/csv",
    )


promo_picker()

step("2", "Future Work ที่จะต่อยอด")
for i in range(0, len(FUTURE_WORK), 4):
    cols = st.columns(4)
    for col, (title, tech, desc) in zip(cols, FUTURE_WORK[i : i + 4]):
        col.markdown(
            f'<div class="card fw"><div class="ttl">{title}</div><span class="tech">{tech}</span><p>{desc}</p></div>',
            unsafe_allow_html=True,
        )
    st.write("")

st.markdown(
    f'<div class="footer">Online Retail II · Data Mining Project · Test เริ่ม {TEST_START} · '
    f"Forecast {FORECAST_MONTH}</div>",
    unsafe_allow_html=True,
)
