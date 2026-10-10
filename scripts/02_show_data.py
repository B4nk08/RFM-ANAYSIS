# Preprocessing 2) แสดงข้อมูล 10 แถว + จำนวนข้อมูล + แต่ละ field เก็บอะไร
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

from common import FIELD_INFO, load_raw
from pretty import banner, show_df, show_kv

banner("PREPROCESSING 2 — แสดงข้อมูล 10 แถว")

df = load_raw()

show_df(df.head(10), title="10 แถวแรก", max_rows=10)

show_kv(
    [
        ("จำนวนแถว", f"{len(df):,}"),
        ("จำนวน field", df.shape[1]),
        ("จำนวนใบเสร็จ", f"{df['Invoice'].nunique():,}"),
        ("จำนวนสินค้า", f"{df['StockCode'].nunique():,}"),
        ("จำนวนลูกค้า", f"{df['Customer ID'].nunique():,}"),
        ("จำนวนประเทศ", df["Country"].nunique()),
    ],
    title="จำนวนข้อมูล",
)

fields = pd.DataFrame(
    {
        "field": list(FIELD_INFO),
        "ค่าที่มีข้อมูล": [int(df[c].notna().sum()) for c in FIELD_INFO],
        "เก็บข้อมูลอะไร": list(FIELD_INFO.values()),
    }
)
show_df(fields, title="แต่ละ field แสดงอะไร")
