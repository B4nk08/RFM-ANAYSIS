"""โค้ดที่ใช้ร่วมกันระหว่างสคริปต์ใน terminal และหน้าเว็บ (app.py)"""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "online_retail_II.csv"
CLEAN_PATH = ROOT / "data" / "cleaned_data.csv"
FIG_DIR = ROOT / "outputs" / "figures"
TABLE_DIR = ROOT / "outputs" / "tables"

DATASET_INFO = [
    ("ชื่อ dataset", "Online Retail II"),
    ("แหล่งที่มา", "UCI Machine Learning Repository (id 502)"),
    ("ไฟล์", "data/online_retail_II.csv"),
    ("ธุรกิจ", "ร้านขายของขวัญออนไลน์ใน UK (ลูกค้าส่วนใหญ่เป็นร้านค้าส่ง)"),
    ("ช่วงเวลา", "1 ธ.ค. 2009 – 9 ธ.ค. 2011"),
    ("1 แถว คือ", "สินค้า 1 รายการในใบเสร็จ 1 ใบ"),
]

FIELD_INFO = {
    "Invoice": "เลขใบเสร็จ (ซ้ำได้ เพราะ 1 ใบมีหลายสินค้า · ขึ้นต้น C = ยกเลิก)",
    "StockCode": "รหัสสินค้า",
    "Description": "ชื่อสินค้า",
    "Quantity": "จำนวนชิ้นที่ซื้อ (ติดลบ = คืนสินค้า)",
    "InvoiceDate": "วันและเวลาที่ซื้อ",
    "Price": "ราคาต่อชิ้น (ปอนด์ £)",
    "Customer ID": "รหัสลูกค้า",
    "Country": "ประเทศของลูกค้า",
}

MISSING_DECISIONS = pd.DataFrame(
    {
        "คอลัมน์": ["Customer ID", "Description", "Quantity ≤ 0", "Price ≤ 0"],
        "ปัญหา": ["ค่าหาย", "ค่าหาย", "คืนสินค้า / ยกเลิก", "ของแถม / ปรับยอด"],
        "วิธีจัดการ": ["ตัดแถวทิ้ง (dropna)", "เติม 'Unknown'", "ตัดแถวทิ้ง", "ตัดแถวทิ้ง"],
        "เหตุผล": [
            "เป็นรหัส เติม Mean/Median/Mode = สร้างลูกค้าปลอม",
            "เป็นข้อความ ไม่ใช้คำนวณ และหายน้อยมาก",
            "ไม่ใช่การขายจริง ทำให้ยอดขายเพี้ยน",
            "ไม่ใช่การขายจริง",
        ],
    }
)

FEATURES = [
    "qty_lag1",
    "qty_lag2",
    "qty_lag3",
    "qty_avg3",
    "qty_last_year",
    "customers_lag1",
    "invoices_lag1",
    "price_lag1",
    "month",
]
FEATURE_INFO = {
    "qty_lag1": "จำนวนที่ขายได้เดือนก่อน",
    "qty_lag2": "จำนวนที่ขายได้ 2 เดือนก่อน",
    "qty_lag3": "จำนวนที่ขายได้ 3 เดือนก่อน",
    "qty_avg3": "ค่าเฉลี่ยยอดขาย 3 เดือนก่อน",
    "qty_last_year": "จำนวนที่ขายได้เดือนเดียวกันปีที่แล้ว",
    "customers_lag1": "จำนวนลูกค้าที่ซื้อเดือนก่อน",
    "invoices_lag1": "จำนวนใบเสร็จเดือนก่อน",
    "price_lag1": "ราคาเฉลี่ยเดือนก่อน",
    "month": "เดือน (1–12) ใช้จับฤดูกาล",
}
HOT_QUANTILE = 0.90
TEST_START = "2011-06"
FORECAST_MONTH = "2011-12"

MONTH_TH = ["ม.ค.", "ก.พ.", "มี.ค.", "เม.ย.", "พ.ค.", "มิ.ย.", "ก.ค.", "ส.ค.", "ก.ย.", "ต.ค.", "พ.ย.", "ธ.ค."]


def month_label(ym: str) -> str:
    y, m = ym.split("-")
    return f"{MONTH_TH[int(m) - 1]} {y}"


# ------------------------------------------------------------
# Preprocessing
# ------------------------------------------------------------
def load_raw() -> pd.DataFrame:
    return pd.read_csv(RAW_PATH, encoding="ISO-8859-1", dtype={"Invoice": str, "StockCode": str})


def dtype_table(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "column": df.columns,
            "dtype": df.dtypes.astype(str).values,
            "ประเภท": ["ตัวเลข" if pd.api.types.is_numeric_dtype(df[c]) else "ข้อความ" for c in df.columns],
            "ตัวอย่าง": [str(df[c].dropna().iloc[0]) for c in df.columns],
            "ค่าไม่ซ้ำ": [df[c].nunique() for c in df.columns],
        }
    )


def missing_table(df: pd.DataFrame) -> pd.DataFrame:
    miss = df.isna().sum()
    return pd.DataFrame(
        {"column": miss.index, "missing": miss.values, "missing_%": (miss.values / len(df) * 100).round(2)}
    )


def outlier_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in ["Quantity", "Price"]:
        s = df[col]
        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        rows.append(
            {
                "column": col,
                "min": s.min(),
                "max": s.max(),
                "IQR_lower": lo,
                "IQR_upper": hi,
                "outliers_IQR": int(((s < lo) | (s > hi)).sum()),
                "ค่า ≤ 0": int((s <= 0).sum()),
            }
        )
    return pd.DataFrame(rows)


def impute_comparison(df: pd.DataFrame) -> pd.DataFrame:
    cid = df["Customer ID"]
    n_miss = int(cid.isna().sum())
    rows = []
    for method, value in [
        ("Min", cid.min()),
        ("Max", cid.max()),
        ("Mean", cid.mean()),
        ("Median", cid.median()),
        ("Mode", cid.mode().iloc[0]),
    ]:
        before = int((cid == value).sum())
        rows.append(
            {
                "วิธี": method,
                "ค่าที่เติม": round(float(value), 2),
                "แถวของ ID นี้ก่อนเติม": before,
                "แถวของ ID นี้หลังเติม": before + n_miss,
                "ผล": "ลูกค้าที่ไม่มีอยู่จริง" if before == 0 else "ลูกค้าคนเดียวยอดพุ่งผิดปกติ",
            }
        )
    rows.append(
        {
            "วิธี": "Drop (ที่เลือกใช้)",
            "ค่าที่เติม": "-",
            "แถวของ ID นี้ก่อนเติม": "-",
            "แถวของ ID นี้หลังเติม": "-",
            "ผล": "ไม่มีลูกค้าปลอม ข้อมูลที่เหลือถูกต้อง",
        }
    )
    out = pd.DataFrame(rows)
    for col in ["ค่าที่เติม", "แถวของ ID นี้ก่อนเติม", "แถวของ ID นี้หลังเติม"]:
        out[col] = [f"{v:,}" if isinstance(v, (int, float)) else v for v in out[col]]
    return out


def clean_data(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    steps = [("ข้อมูลดิบ", len(raw))]
    df = raw.copy()
    df["Description"] = df["Description"].fillna("Unknown")
    steps.append(("เติม Description ที่หาย = 'Unknown'", len(df)))
    df = df.dropna(subset=["Customer ID"])
    steps.append(("ตัดแถวที่ไม่มี Customer ID", len(df)))
    df = df[df["Quantity"] > 0]
    steps.append(("ตัด Quantity ≤ 0", len(df)))
    df = df[df["Price"] > 0].copy()
    steps.append(("ตัด Price ≤ 0", len(df)))

    df["Customer ID"] = df["Customer ID"].astype(int)
    df["StockCode"] = df["StockCode"].str.strip()
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["TotalPrice"] = df["Quantity"] * df["Price"]
    df = df.reset_index(drop=True)

    s = pd.DataFrame(steps, columns=["ขั้นตอน", "แถวคงเหลือ"])
    s["ตัดออก"] = (s["แถวคงเหลือ"].shift(1) - s["แถวคงเหลือ"]).fillna(0).astype(int)
    return df, s


def load_clean() -> pd.DataFrame:
    if not CLEAN_PATH.exists():
        clean, _ = clean_data(load_raw())
        clean.to_csv(CLEAN_PATH, index=False)
        return clean
    df = pd.read_csv(CLEAN_PATH, dtype={"Invoice": str, "StockCode": str}, parse_dates=["InvoiceDate"])
    if "TotalPrice" not in df.columns:
        df["TotalPrice"] = df["Quantity"] * df["Price"]
    return df


# ------------------------------------------------------------
# Figures (ชื่อกราฟเป็นภาษาอังกฤษ เพราะฟอนต์ matplotlib ไม่มีภาษาไทย)
# ------------------------------------------------------------
PALETTE = {"main": "#4f46e5", "accent": "#06b6d4", "warn": "#f43f5e", "ok": "#10b981"}


def save_fig(fig, name: str) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return path.relative_to(ROOT)


def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25)


def fig_missing(raw: pd.DataFrame):
    m = missing_table(raw).sort_values("missing_%")
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = [PALETTE["warn"] if v > 0 else "#cbd5e1" for v in m["missing_%"]]
    bars = ax.barh(m["column"], m["missing_%"], color=colors)
    ax.bar_label(bars, labels=[f"{v:.2f}%" for v in m["missing_%"]], padding=4, fontsize=9)
    ax.set_xlabel("Missing (%)")
    ax.set_title("Missing values per column", fontweight="bold")
    ax.set_xlim(0, m["missing_%"].max() * 1.25)
    _style(ax)
    fig.tight_layout()
    return fig


def fig_boxplot(raw: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    for ax, col, color in zip(axes, ["Quantity", "Price"], [PALETTE["main"], PALETTE["accent"]]):
        ax.boxplot(
            raw[col],
            patch_artist=True,
            boxprops={"facecolor": color, "alpha": 0.4},
            flierprops={"marker": "o", "markersize": 3, "alpha": 0.3, "markerfacecolor": color},
        )
        ax.set_title(f"{col} (raw)", fontweight="bold")
        ax.set_xticks([])
        _style(ax)
    fig.suptitle("Outliers (boxplot)", fontweight="bold")
    fig.tight_layout()
    return fig


def fig_scatter(clean: pd.DataFrame, n: int = 3000):
    s = clean.sample(min(n, len(clean)), random_state=42)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    axes[0].scatter(s["Quantity"], s["Price"], s=10, alpha=0.45, color=PALETTE["main"])
    axes[0].set_title("Normal scale", fontweight="bold")
    axes[1].scatter(s["Quantity"], s["Price"], s=10, alpha=0.45, color=PALETTE["accent"])
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    plain = FuncFormatter(lambda v, _: f"{v:g}")
    axes[1].xaxis.set_major_formatter(plain)
    axes[1].yaxis.set_major_formatter(plain)
    axes[1].set_title("Log scale", fontweight="bold")
    for ax in axes:
        ax.set_xlabel("Quantity")
        ax.set_ylabel("Price (£)")
        _style(ax)
    fig.suptitle(f"Quantity vs Price ({len(s):,} sampled rows)", fontweight="bold")
    fig.tight_layout()
    return fig


def fig_heatmap(clean: pd.DataFrame):
    corr = clean[["Quantity", "Price", "TotalPrice"]].corr()
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, vmin=-1, vmax=1, square=True, ax=ax,
                linewidths=1, linecolor="white", annot_kws={"fontsize": 12, "fontweight": "bold"})
    ax.set_title("Correlation heatmap", fontweight="bold")
    fig.tight_layout()
    return fig, corr


def fig_confusion(cm: np.ndarray):
    fig, ax = plt.subplots(figsize=(5, 4.2))
    sns.heatmap(cm, annot=True, fmt=",d", cmap="Purples", cbar=False, ax=ax, linewidths=1, linecolor="white",
                annot_kws={"fontsize": 13, "fontweight": "bold"},
                xticklabels=["Pred: not hot", "Pred: hot"], yticklabels=["Actual: not hot", "Actual: hot"])
    ax.set_title("Confusion matrix (test set)", fontweight="bold")
    fig.tight_layout()
    return fig


def fig_feature_importance(model, features=FEATURES):
    imp = pd.Series(model.feature_importances_, index=features).sort_values()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(imp.index, imp.values, color=PALETTE["ok"])
    ax.set_title("Feature importance (Gradient Boosting)", fontweight="bold")
    _style(ax)
    fig.tight_layout()
    return fig


# ------------------------------------------------------------
# Model: ทำนายว่าสินค้าจะ "ขายดี" ในเดือนนั้นไหม
# ------------------------------------------------------------
def build_product_month(clean: pd.DataFrame) -> pd.DataFrame:
    df = clean.copy()
    df["YearMonth"] = df["InvoiceDate"].dt.to_period("M")
    agg = df.groupby(["StockCode", "YearMonth"]).agg(
        qty=("Quantity", "sum"),
        customers=("Customer ID", "nunique"),
        invoices=("Invoice", "nunique"),
        price=("Price", "mean"),
    )

    months = pd.period_range(df["YearMonth"].min(), df["YearMonth"].max(), freq="M")
    grid = pd.MultiIndex.from_product([agg.index.levels[0], months], names=["StockCode", "YearMonth"])
    pm = agg.reindex(grid).reset_index()
    pm[["qty", "customers", "invoices"]] = pm[["qty", "customers", "invoices"]].fillna(0)

    g = pm.groupby("StockCode")
    pm["qty_lag1"] = g["qty"].shift(1)
    pm["qty_lag2"] = g["qty"].shift(2)
    pm["qty_lag3"] = g["qty"].shift(3)
    pm["qty_avg3"] = pm[["qty_lag1", "qty_lag2", "qty_lag3"]].mean(axis=1)
    pm["qty_last_year"] = g["qty"].shift(12).fillna(0)
    pm["customers_lag1"] = g["customers"].shift(1)
    pm["invoices_lag1"] = g["invoices"].shift(1)
    pm["price_lag1"] = g["price"].shift(1)
    pm["month"] = pm["YearMonth"].dt.month

    pm = pm.dropna(subset=["qty_lag3"])
    pm = pm[(pm["qty_lag1"] > 0) | (pm["qty_lag2"] > 0)]
    pm["price_lag1"] = pm["price_lag1"].fillna(pm.groupby("StockCode")["price"].transform("mean"))
    pm = pm.dropna(subset=["price_lag1"])

    sold = pm[pm["qty"] > 0]
    threshold = sold.groupby("YearMonth")["qty"].quantile(HOT_QUANTILE)
    pm["hot_threshold"] = pm["YearMonth"].map(threshold)
    pm["is_hot"] = (pm["qty"] >= pm["hot_threshold"]).astype(int)

    pm["YearMonth"] = pm["YearMonth"].astype(str)
    names = df.groupby("StockCode")["Description"].agg(lambda s: s.mode().iloc[0])
    pm["Description"] = pm["StockCode"].map(names)
    return pm.reset_index(drop=True)


MODEL_NAME = "Gradient Boosting"

MODEL_REASON = (
    "ใช้ Gradient Boosting เพราะข้อมูลยอดขายไม่สัมพันธ์แบบเส้นตรงและมีค่าสุดโต่งเยอะ "
    "โมเดลนี้รับมือได้ดีโดยไม่ต้อง scale ข้อมูล และให้ผลเป็น % โอกาสขายดี "
    "ที่เอามาเรียงอันดับสินค้าเพื่อจัดโปรได้"
)


def make_model() -> GradientBoostingClassifier:
    return GradientBoostingClassifier(random_state=42)


def split_data(pm: pd.DataFrame):
    labeled = pm[pm["YearMonth"] < FORECAST_MONTH]
    train = labeled[labeled["YearMonth"] < TEST_START]
    test = labeled[labeled["YearMonth"] >= TEST_START]
    return labeled, train, test


def train_model(pm: pd.DataFrame):
    labeled, train, test = split_data(pm)

    t0 = time.time()
    eval_model = make_model().fit(train[FEATURES], train["is_hot"])
    train_sec = time.time() - t0
    pred = eval_model.predict(test[FEATURES])
    proba = eval_model.predict_proba(test[FEATURES])[:, 1]
    y = test["is_hot"]
    metrics = {
        "Accuracy": accuracy_score(y, pred),
        "Precision": precision_score(y, pred, zero_division=0),
        "Recall": recall_score(y, pred, zero_division=0),
        "F1": f1_score(y, pred, zero_division=0),
        "ROC_AUC": roc_auc_score(y, proba),
    }

    final_model = make_model().fit(labeled[FEATURES], labeled["is_hot"])

    info = {
        "n_rows": len(pm),
        "n_labeled": len(labeled),
        "n_train": len(train),
        "n_test": len(test),
        "n_forecast": len(pm) - len(labeled),
        "train_months": f"{train['YearMonth'].min()} → {train['YearMonth'].max()}",
        "test_months": f"{test['YearMonth'].min()} → {test['YearMonth'].max()}",
        "all_months": f"{labeled['YearMonth'].min()} → {labeled['YearMonth'].max()}",
        "hot_rate": pm["is_hot"].mean(),
        "train_sec": train_sec,
        "cm": confusion_matrix(y, pred),
    }
    return metrics, eval_model, final_model, info


def predict_hot(pm: pd.DataFrame, eval_model, final_model) -> pd.DataFrame:
    out = pm.copy()
    out["hot_prob"] = eval_model.predict_proba(out[FEATURES])[:, 1]
    is_forecast = out["YearMonth"] >= FORECAST_MONTH
    out.loc[is_forecast, "hot_prob"] = final_model.predict_proba(out.loc[is_forecast, FEATURES])[:, 1]
    out["period"] = np.where(
        out["YearMonth"] >= FORECAST_MONTH,
        "Forecast",
        np.where(out["YearMonth"] >= TEST_START, "Test", "Train"),
    )
    return out


def discount_for(rank: int) -> str:
    if rank <= 3:
        return "15%"
    if rank <= 6:
        return "10%"
    return "5%"


def promo_for_month(pred: pd.DataFrame, ym: str, top_n: int = 10, show_actual: bool = True) -> pd.DataFrame:
    m = pred[pred["YearMonth"] == ym].nlargest(top_n, "hot_prob")
    ranks = range(1, len(m) + 1)
    out = pd.DataFrame(
        {
            "อันดับ": ranks,
            "รหัสสินค้า": m["StockCode"].values,
            "สินค้า": m["Description"].values,
            "ขายเดือนก่อน (ชิ้น)": m["qty_lag1"].astype(int).values,
            "โอกาสขายดี (%)": (m["hot_prob"] * 100).round(1).values,
            "ส่วนลดแนะนำ": [discount_for(r) for r in ranks],
        }
    )
    if show_actual and ym < FORECAST_MONTH:
        out["ขายจริงเดือนนี้ (ชิ้น)"] = m["qty"].astype(int).values
        out["ขายดีจริง?"] = np.where(m["is_hot"].values == 1, "✔", "✘")
    return out


def hit_rate_at_k(pred: pd.DataFrame, months: list[str], k: int = 10) -> float:
    hits = [pred[pred["YearMonth"] == ym].nlargest(k, "hot_prob")["is_hot"].mean() for ym in months]
    return float(np.mean(hits))


def promo_all_months(pred: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    frames = []
    for ym in sorted(pred["YearMonth"].unique()):
        t = promo_for_month(pred, ym, top_n)
        t.insert(0, "เดือน", ym)
        frames.append(t)
    return pd.concat(frames, ignore_index=True)


FUTURE_WORK = [
    ("RFM Customer Segmentation", "RFM score + K-Means",
     "แบ่งลูกค้าเป็น VIP / Loyal / At-risk / Lost แล้วส่งโปรให้ตรงกลุ่ม"),
    ("Market Basket / ซื้อคู่", "Apriori / FP-Growth",
     "หาสินค้าที่ซื้อด้วยกันบ่อย ทำโปร 'ซื้อคู่ลด' ต่อจากโปรรายเดือน"),
    ("Sales Forecasting", "Prophet / ARIMA / LSTM",
     "พยากรณ์จำนวนชิ้นที่จะขายได้ (ไม่ใช่แค่ขายดีไหม) เพื่อวางแผนสต็อก"),
    ("Churn Prediction", "Random Forest / XGBoost",
     "ทำนายลูกค้าที่กำลังจะเลิกซื้อ แล้วส่งคูปองดึงกลับ"),
    ("Customer Lifetime Value", "BG/NBD + Gamma-Gamma",
     "ประเมินมูลค่าลูกค้าในอนาคต เลือกลงทุนกับลูกค้าที่คุ้มค่า"),
    ("A/B Testing โปรโมชัน", "Hypothesis testing",
     "ทดลองให้ส่วนลดจริงกับลูกค้า 2 กลุ่ม วัดว่ายอดขายเพิ่มจริงไหม"),
    ("ปรับจูนโมเดล", "GridSearchCV / XGBoost / LightGBM",
     "จูน hyperparameter ของ Gradient Boosting, ลองเทียบกับโมเดลอื่น และเพิ่ม feature เช่น วันหยุด, ประเทศ"),
    ("Recommendation System", "Collaborative Filtering",
     "แนะนำสินค้าเฉพาะลูกค้าแต่ละคน จากประวัติการซื้อของลูกค้าที่คล้ายกัน"),
    ("Returns Analysis", "Classification",
     "ใช้แถวคืนสินค้า (Quantity ติดลบ) ที่ตัดทิ้ง มาหาว่าสินค้า / ลูกค้าแบบไหนคืนบ่อย"),
    ("Country Analysis", "Clustering",
     "จัดกลุ่มประเทศตามพฤติกรรมการซื้อ หาตลาดต่างประเทศที่มีศักยภาพนอกจาก UK"),
]
