# ============================================================
# ข้อ 1) โหลดข้อมูล สร้าง DataFrame และ print head 10 แถวแรก
# ============================================================
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pretty import banner, ok, show_df, show_kv

banner("STEP 1: Load DataFrame + head(10)")

df = pd.read_csv("data/online_retail_II.csv", encoding="ISO-8859-1")

ok(f"Loaded: [bold]online_retail_II.csv[/bold]")
show_kv(
    [
        ("rows", f"{df.shape[0]:,}"),
        ("columns", df.shape[1]),
        ("column names", ", ".join(map(str, df.columns))),
    ],
    title="Dataset overview",
)
show_df(df, title="Head — 10 rows", max_rows=10)
