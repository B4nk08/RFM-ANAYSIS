# Preprocessing 6) สรุป data ที่ได้ว่าเหลือเท่าไหร่
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import load_clean, load_raw
from pretty import banner, ok, show_df, show_kv

banner("PREPROCESSING 6 — สรุปข้อมูลหลัง Clean")

raw = load_raw()
clean = load_clean()

show_kv(
    [
        ("แถวก่อน clean", f"{len(raw):,}"),
        ("แถวหลัง clean", f"{len(clean):,}"),
        ("ตัดออก", f"{len(raw) - len(clean):,} ({(1 - len(clean) / len(raw)) * 100:.1f}%)"),
        ("เหลือ", f"{len(clean) / len(raw) * 100:.1f}%"),
        ("จำนวน field", f"{clean.shape[1]} (เพิ่ม TotalPrice = Quantity × Price)"),
        ("ใบเสร็จ", f"{clean['Invoice'].nunique():,}"),
        ("ลูกค้า", f"{clean['Customer ID'].nunique():,}"),
        ("สินค้า", f"{clean['StockCode'].nunique():,}"),
        ("ช่วงเวลา", f"{clean['InvoiceDate'].min():%Y-%m-%d} → {clean['InvoiceDate'].max():%Y-%m-%d}"),
        ("Missing ที่เหลือ", int(clean.isna().sum().sum())),
    ],
    title="สรุป",
)

show_df(clean.head(10), title="ข้อมูลหลัง clean (10 แถวแรก)", max_rows=10)
ok("ข้อมูลพร้อมใช้ทำ Visualization และ Model")
