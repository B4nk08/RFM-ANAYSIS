# ============================================================
# ดูข้อมูลที่ clean แล้ว: data/cleaned_data.csv
# ต้องรันข้อ 4 ก่อน (scripts/4_handle_missing.py)
# ============================================================
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pretty import banner, ok, show_df, show_kv, warn

banner("Cleaned Data — data/cleaned_data.csv")

path = Path("data/cleaned_data.csv")
if not path.exists():
    warn("ยังไม่มีไฟล์ — รันก่อน: python3 scripts/4_handle_missing.py")
    raise SystemExit(1)

df = pd.read_csv(path)

show_kv(
    [
        ("file", str(path)),
        ("rows", f"{len(df):,}"),
        ("columns", df.shape[1]),
        ("column names", ", ".join(map(str, df.columns))),
        ("customers", f"{df['Customer ID'].nunique():,}"),
    ],
    title="Overview",
)

show_df(df, title="Head — 10 rows", max_rows=10)
ok("นี่คือข้อมูลหลัง clean แล้ว ใช้ต่อข้อ 5–6 / RFM ได้")
