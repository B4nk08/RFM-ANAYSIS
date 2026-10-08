# ============================================================
# สร้างรายงาน HTML รวมผลทุกข้อ → outputs/report.html
# ============================================================
import base64
import os
import webbrowser
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
os.makedirs("outputs/figures", exist_ok=True)

REPORT = Path("outputs/report.html")
RAW = Path("data/online_retail_II.csv")
CLEAN = Path("data/cleaned_data.csv")


def img_tag(path: Path, alt: str = "") -> str:
    if not path.exists():
        return f"<p>ยังไม่มีไฟล์กราฟ: {path}</p>"
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    return f'<img class="fig" src="data:image/png;base64,{b64}" alt="{alt}">'


def _cell(val) -> str:
    if pd.isna(val):
        return "—"
    if isinstance(val, (np.integer, int)) and not isinstance(val, bool):
        return f"{int(val):,}"
    if isinstance(val, (np.floating, float)):
        if float(val).is_integer() and abs(val) < 1e12:
            return f"{int(val):,}"
        return f"{val:,.2f}"
    return str(val)


def tbl(df: pd.DataFrame, n: int = 10) -> str:
    """ตาราง HTML — ตัวเลขอ่านง่าย ไม่ scientific · กว้างเต็มกล่อง"""
    view = df.head(n).copy()
    cols = list(view.columns)
    rows_html = []
    for _, row in view.iterrows():
        tds = "".join(f"<td>{_cell(row[c])}</td>" for c in cols)
        rows_html.append(f"<tr>{tds}</tr>")

    thead = "".join(f"<th>{c}</th>" for c in cols)
    body = "\n".join(rows_html)
    return f'<div class="scroll"><table class="tbl"><thead><tr>{thead}</tr></thead><tbody>{body}</tbody></table></div>'


def ensure_figures(df: pd.DataFrame, cleaned: pd.DataFrame | None) -> None:
    miss_path = Path("outputs/figures/2_missing_values.png")
    box_path = Path("outputs/figures/2_outliers_boxplot.png")
    if not miss_path.exists() or not box_path.exists():
        pct = (df.isnull().sum() / len(df)) * 100
        summary = pd.DataFrame({"Missing %": pct.round(2)})
        summary = summary[summary["Missing %"] > 0]
        plt.figure(figsize=(8, 4.2), facecolor="white")
        bars = plt.bar(summary.index, summary["Missing %"], color="#0f766e")
        plt.ylabel("% Missing")
        plt.title("Missing Values by Column")
        plt.xticks(rotation=20, ha="right")
        for bar in bars:
            h = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, h + 0.4, f"{h:.1f}%", ha="center", fontsize=10)
        plt.tight_layout()
        plt.savefig(miss_path, dpi=150, bbox_inches="tight")
        plt.close()

        fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), facecolor="white")
        sns.boxplot(y=df["Quantity"], ax=axes[0], color="#99f6e4")
        axes[0].set_title("Quantity — Outliers")
        sns.boxplot(y=df["Price"], ax=axes[1], color="#fecdd3")
        axes[1].set_title("Price — Outliers")
        plt.tight_layout()
        plt.savefig(box_path, dpi=150, bbox_inches="tight")
        plt.close()

    if cleaned is None:
        return

    scatter = Path("outputs/figures/5_scatter_quantity_price.png")
    if not scatter.exists():
        sample = cleaned.sample(min(2000, len(cleaned)), random_state=42)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), facecolor="white")
        sns.scatterplot(x="Quantity", y="Price", data=sample, ax=axes[0], s=18, alpha=0.45, color="#0f766e")
        axes[0].set_title("Quantity vs Price")
        sns.scatterplot(x="Quantity", y="Price", data=sample, ax=axes[1], s=18, alpha=0.45, color="#0284c7")
        axes[1].set_xscale("log")
        axes[1].set_yscale("log")
        axes[1].set_title("Quantity vs Price (Log)")
        plt.tight_layout()
        plt.savefig(scatter, dpi=150, bbox_inches="tight")
        plt.close()

    heat = Path("outputs/figures/6_correlation_heatmap.png")
    if not heat.exists():
        work = cleaned.copy()
        if "TotalPrice" not in work.columns:
            work["TotalPrice"] = work["Quantity"] * work["Price"]
        corr = work[["Quantity", "Price", "TotalPrice", "Customer ID"]].corr()
        plt.figure(figsize=(6.5, 5.5), facecolor="white")
        sns.heatmap(corr, annot=True, cmap="RdBu_r", fmt=".2f", center=0, square=True)
        plt.title("Correlation Heatmap")
        plt.tight_layout()
        plt.savefig(heat, dpi=150, bbox_inches="tight")
        plt.close()


print("กำลังสร้างรายงาน HTML...")

df = pd.read_csv(RAW, encoding="ISO-8859-1")
cleaned = pd.read_csv(CLEAN) if CLEAN.exists() else None
ensure_figures(df, cleaned)

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
text_cols = df.select_dtypes(include=["object"]).columns.tolist()
dtypes_tbl = pd.DataFrame({"column": df.columns, "dtype": df.dtypes.astype(str).values})

missing = df.isnull().sum()
miss_tbl = pd.DataFrame(
    {
        "column": missing.index,
        "Missing Count": missing.values,
        "Missing %": (missing / len(df) * 100).round(2),
    }
)
miss_tbl = miss_tbl[miss_tbl["Missing Count"] > 0].sort_values("Missing Count", ascending=False)

outlier = pd.DataFrame(
    {
        "check": ["Quantity < 0", "Price <= 0", "Customer ID missing", "Description missing"],
        "count": [
            int((df["Quantity"] < 0).sum()),
            int((df["Price"] <= 0).sum()),
            int(df["Customer ID"].isnull().sum()),
            int(df["Description"].isnull().sum()),
        ],
    }
)

stats_view = (
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
skew = df[numeric_cols].skew().round(2).reset_index()
skew.columns = ["feature", "skewness"]
cleaned_count = f"{len(cleaned):,}" if cleaned is not None else "—"

if cleaned is not None:
    corr_work = cleaned.copy()
    if "TotalPrice" not in corr_work.columns:
        corr_work["TotalPrice"] = corr_work["Quantity"] * corr_work["Price"]
    corr = corr_work[["Quantity", "Price", "TotalPrice", "Customer ID"]].corr().round(3)
    corr_view = corr.reset_index().rename(columns={"index": "feature"})
    q_p_corr = cleaned["Quantity"].corr(cleaned["Price"])
    clean_block = f"""
<section id="s4">
  <div class="sec-head"><div class="badge">04</div><div>
    <h2>Handle Missing</h2>
    <p>Customer ID drop · Description → Unknown · Quantity/Price ≤ 0 drop</p>
  </div></div>
  <div class="stat-row">
    <div class="stat"><span>Before</span><b>{len(df):,}</b></div>
    <div class="stat"><span>After</span><b>{len(cleaned):,}</b></div>
    <div class="stat"><span>Removed</span><b>{len(df) - len(cleaned):,}</b></div>
  </div>
  <h3>cleaned_data — head(10)</h3>
  {tbl(cleaned, 10)}
</section>

<section id="s5">
  <div class="sec-head"><div class="badge">05</div><div>
    <h2>Scatter — Quantity vs Price</h2>
    <p>Correlation ≈ {q_p_corr:.3f}</p>
  </div></div>
  {img_tag(Path("outputs/figures/5_scatter_quantity_price.png"), "scatter")}
</section>

<section id="s6">
  <div class="sec-head"><div class="badge">06</div><div>
    <h2>Correlation Heatmap</h2>
    <p>Numeric features after cleaning</p>
  </div></div>
  <div class="split">
    <div>{img_tag(Path("outputs/figures/6_correlation_heatmap.png"), "heatmap")}</div>
    <div>
      <h3>Correlation matrix</h3>
      {tbl(corr_view, 20)}
    </div>
  </div>
</section>
"""
else:
    clean_block = """
<section id="s4">
  <div class="sec-head"><div class="badge">04</div><div>
    <h2>Handle Missing</h2>
    <p>Run <code>python3 scripts/4_handle_missing.py</code> first</p>
  </div></div>
</section>
"""

html = f"""<!DOCTYPE html>
<html lang="th">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>RFM Analysis Report</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Sora:wght@500;600;700&family=IBM+Plex+Sans:wght@400;500;600&display=swap" rel="stylesheet">
<style>
:root {{
  --ink: #0b1220;
  --muted: #5b6578;
  --line: #e6ebf2;
  --bg: #eef2f7;
  --paper: #ffffff;
  --accent: #0f766e;
  --accent-2: #0e7490;
  --soft: #e6fffa;
}}
* {{ box-sizing: border-box; }}
html {{ scroll-behavior: smooth; }}
body {{
  margin: 0;
  font-family: "IBM Plex Sans", system-ui, sans-serif;
  color: var(--ink);
  background:
    radial-gradient(900px 380px at 8% -5%, #d1fae5 0%, transparent 55%),
    radial-gradient(700px 320px at 100% 0%, #cffafe 0%, transparent 50%),
    var(--bg);
  line-height: 1.5;
}}
.wrap {{ max-width: 1120px; margin: 0 auto; padding: 20px 16px 64px; }}

.nav {{
  position: sticky; top: 0; z-index: 20;
  margin: -20px -16px 18px; padding: 10px 16px;
  background: rgba(238,242,247,.9);
  backdrop-filter: blur(12px);
  border-bottom: 1px solid var(--line);
}}
.nav-inner {{
  max-width: 1120px; margin: 0 auto;
  display: flex; gap: 4px; flex-wrap: wrap; align-items: center;
}}
.brand {{
  font-family: Sora, sans-serif; font-weight: 700; font-size: 14px;
  margin-right: 10px; color: var(--ink);
}}
.nav a {{
  text-decoration: none; color: var(--muted); font-size: 12.5px; font-weight: 600;
  padding: 6px 11px; border-radius: 999px;
}}
.nav a:hover {{ background: var(--soft); color: var(--accent); }}

.hero {{
  background: linear-gradient(135deg, #0b1220 0%, #115e59 48%, #0e7490 100%);
  color: #fff; border-radius: 20px; padding: 32px 28px 28px;
  box-shadow: 0 16px 40px rgba(11, 18, 32, .18);
}}
.hero .kicker {{
  margin: 0 0 8px; font-size: 11px; letter-spacing: .16em; text-transform: uppercase;
  opacity: .7; font-weight: 600;
}}
.hero h1 {{
  font-family: Sora, sans-serif; margin: 0 0 8px;
  font-size: clamp(26px, 4.5vw, 36px); line-height: 1.12; font-weight: 700;
}}
.hero p {{ margin: 0; opacity: .88; max-width: 48ch; font-size: 15px; }}
.metrics {{
  display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin-top: 22px;
}}
@media (max-width: 700px) {{ .metrics {{ grid-template-columns: 1fr; }} }}
.metric {{
  background: rgba(255,255,255,.1); border: 1px solid rgba(255,255,255,.14);
  border-radius: 14px; padding: 12px 14px;
}}
.metric span {{ display: block; font-size: 11px; opacity: .72; margin-bottom: 2px; }}
.metric b {{ font-family: Sora, sans-serif; font-size: 22px; font-weight: 700; }}

section {{
  margin-top: 16px; background: var(--paper); border: 1px solid var(--line);
  border-radius: 18px; padding: 20px 18px 18px;
  box-shadow: 0 6px 20px rgba(11, 18, 32, .04);
}}
.sec-head {{ display: flex; gap: 12px; align-items: flex-start; margin-bottom: 14px; }}
.badge {{
  width: 38px; height: 38px; border-radius: 11px; flex: 0 0 auto;
  background: var(--soft); color: var(--accent);
  display: grid; place-items: center;
  font-family: Sora, sans-serif; font-weight: 700; font-size: 13px;
}}
.sec-head h2 {{ font-family: Sora, sans-serif; margin: 0 0 2px; font-size: 18px; }}
.sec-head p {{ margin: 0; color: var(--muted); font-size: 13px; }}
h3 {{
  font-family: Sora, sans-serif; font-size: 13px; margin: 14px 0 8px;
  color: var(--muted); text-transform: uppercase; letter-spacing: .04em;
}}

.scroll {{
  overflow-x: auto; border: 1px solid var(--line); border-radius: 12px;
  background: #fff;
}}
.tbl {{
  border-collapse: collapse; width: 100%;
  font-size: 12.5px; table-layout: auto;
}}
.tbl th {{
  background: #0b1220; color: #fff; text-align: left;
  padding: 8px 12px; font-weight: 600; white-space: nowrap;
  position: sticky; top: 0;
}}
.tbl td {{
  padding: 7px 12px; border-bottom: 1px solid var(--line);
  white-space: nowrap; vertical-align: middle;
}}
.tbl tbody tr:nth-child(even) td {{ background: #f7fafc; }}
.tbl tbody tr:hover td {{ background: #ecfeff; }}

.fig {{
  width: 100%; max-height: 420px; object-fit: contain;
  border-radius: 12px; border: 1px solid var(--line);
  background: #fafbfc; display: block; margin: 4px 0;
}}

.trio {{
  display: grid; grid-template-columns: 1.1fr 1fr .9fr; gap: 12px;
}}
@media (max-width: 900px) {{ .trio {{ grid-template-columns: 1fr; }} }}
.card {{
  border: 1px solid var(--line); border-radius: 12px; padding: 10px;
  background: #fafbfc; min-height: 0;
}}
.card h3 {{ margin-top: 0; }}
.card .scroll {{ border: none; border-radius: 8px; }}

.split {{
  display: grid; grid-template-columns: 1.15fr .85fr; gap: 14px; align-items: start;
}}
@media (max-width: 900px) {{ .split {{ grid-template-columns: 1fr; }} }}

.charts {{
  display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-top: 8px;
}}
@media (max-width: 800px) {{ .charts {{ grid-template-columns: 1fr; }} }}

.stat-row {{
  display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin: 4px 0 12px;
}}
.stat {{
  background: #f0fdfa; border: 1px solid #99f6e4; border-radius: 12px;
  padding: 10px 12px;
}}
.stat span {{ display: block; font-size: 11px; color: var(--muted); }}
.stat b {{ font-family: Sora, sans-serif; font-size: 18px; color: var(--accent); }}

.chips {{ display: flex; flex-wrap: wrap; gap: 6px; margin: 0 0 12px; }}
.chip {{
  font-size: 12px; font-weight: 600; padding: 5px 10px; border-radius: 999px;
  background: #f1f5f9; border: 1px solid var(--line); color: var(--muted);
}}
.chip b {{ color: var(--ink); }}
code {{
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  background: #e8eef5; padding: 1px 6px; border-radius: 5px; font-size: .9em;
}}
footer {{ margin-top: 22px; text-align: center; color: var(--muted); font-size: 12px; }}
</style>
</head>
<body>
<div class="wrap">

<nav class="nav"><div class="nav-inner">
  <span class="brand">RFM Report</span>
  <a href="#s1">1 Load</a><a href="#s2">2 Explore</a><a href="#s3">3 Stats</a>
  <a href="#s4">4 Missing</a><a href="#s5">5 Scatter</a><a href="#s6">6 Heatmap</a>
</div></nav>

<header class="hero">
  <p class="kicker">Data Mining · Online Retail II</p>
  <h1>RFM Analysis Report</h1>
  <p>ตาราง + กราฟครบ 6 ขั้นตอน สำหรับพรีเซนต์</p>
  <div class="metrics">
    <div class="metric"><span>Transactions</span><b>{len(df):,}</b></div>
    <div class="metric"><span>Columns</span><b>{df.shape[1]}</b></div>
    <div class="metric"><span>Cleaned rows</span><b>{cleaned_count}</b></div>
  </div>
</header>

<section id="s1">
  <div class="sec-head"><div class="badge">01</div><div>
    <h2>Load Data</h2>
    <p><code>{RAW.name}</code> — head(10)</p>
  </div></div>
  {tbl(df, 10)}
</section>

<section id="s2">
  <div class="sec-head"><div class="badge">02</div><div>
    <h2>Explore</h2>
    <p>dtypes · missing · outliers</p>
  </div></div>
  <div class="chips">
    <span class="chip">numeric <b>{len(numeric_cols)}</b> — {", ".join(numeric_cols)}</span>
    <span class="chip">text <b>{len(text_cols)}</b></span>
  </div>
  <div class="trio">
    <div class="card"><h3>dtypes</h3>{tbl(dtypes_tbl, 20)}</div>
    <div class="card"><h3>Missing</h3>{tbl(miss_tbl, 20)}</div>
    <div class="card"><h3>Outliers</h3>{tbl(outlier, 10)}</div>
  </div>
  <div class="charts">
    <div>
      <h3>Missing %</h3>
      {img_tag(Path("outputs/figures/2_missing_values.png"), "missing")}
    </div>
    <div>
      <h3>Boxplot</h3>
      {img_tag(Path("outputs/figures/2_outliers_boxplot.png"), "boxplot")}
    </div>
  </div>
</section>

<section id="s3">
  <div class="sec-head"><div class="badge">03</div><div>
    <h2>Statistics</h2>
    <p>Min / Max / Mean / Median / Mode + skewness</p>
  </div></div>
  {tbl(stats_view, 20)}
  <h3>Skewness</h3>
  {tbl(skew, 20)}
</section>

{clean_block}

<footer>scripts/7_make_report.py</footer>
</div>
</body>
</html>
"""

REPORT.write_text(html, encoding="utf-8")
print(f"✅ สร้างรายงานแล้ว: {REPORT.resolve()}")
webbrowser.open(REPORT.resolve().as_uri())
