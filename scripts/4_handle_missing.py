# ============================================================
# ข้อ 4) จัดการ Missing Value → data/cleaned_data.csv
# ============================================================
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pretty import banner, ok, show_df, show_kv, warn

banner("STEP 4: Handle Missing Values")

df = pd.read_csv("data/online_retail_II.csv", encoding="ISO-8859-1")

show_kv(
    [
        ("คอลัมน์ที่มี missing", ", ".join(df.columns[df.isnull().any()].tolist())),
        (
            "Customer ID missing",
            f"{df['Customer ID'].isnull().sum():,} ({df['Customer ID'].isnull().sum() / len(df) * 100:.1f}%)",
        ),
        ("Description missing", f"{df['Description'].isnull().sum():,}"),
    ],
    title="Before clean",
)

warn("Customer ID ไม่ควร impute (จะได้ ID ปลอม) → ตัดแถวทิ้ง")
ok("Description เติมด้วย 'Unknown'")

df_cleaned = df.copy()
df_cleaned["Description"] = df_cleaned["Description"].fillna("Unknown")
df_cleaned = df_cleaned.dropna(subset=["Customer ID"])

before = len(df_cleaned)
df_cleaned = df_cleaned[(df_cleaned["Quantity"] > 0) & (df_cleaned["Price"] > 0)]
after = len(df_cleaned)

decision = pd.DataFrame(
    {
        "Feature": ["Customer ID", "Description", "Quantity/Price <= 0"],
        "Method": ["ตัดแถวทิ้ง", "เติม Unknown", "ตัดแถวทิ้ง"],
        "Reason": [
            "เป็น ID เฉพาะตัว / จำเป็นต่อ RFM",
            "ไม่กระทบการคำนวณตัวเลข",
            "คืนสินค้า / ข้อมูลผิดพลาด",
        ],
    }
)
show_df(decision, title="สรุปการตัดสินใจ", max_rows=10)

show_kv(
    [
        ("ก่อน clean", f"{len(df):,}"),
        ("หลัง clean", f"{len(df_cleaned):,}"),
        ("ตัด Quantity/Price <= 0", f"{before - after:,}"),
        ("ตัดทั้งหมด", f"{len(df) - len(df_cleaned):,}"),
    ],
    title="After clean",
)

miss_after = df_cleaned.isnull().sum().reset_index()
miss_after.columns = ["column", "missing"]
show_df(miss_after, title="Missing หลัง clean", max_rows=20)

df_cleaned.to_csv("data/cleaned_data.csv", index=False)
ok("บันทึก data/cleaned_data.csv แล้ว")
