# Preprocessing 1) เตรียม data — data ชื่ออะไร มาจากไหน
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import DATASET_INFO, RAW_PATH, load_raw
from pretty import banner, ok, show_kv

banner("PREPROCESSING 1 — เตรียมข้อมูล (Dataset)")

show_kv(DATASET_INFO, title="ข้อมูลที่ใช้")

size_mb = RAW_PATH.stat().st_size / 1024**2
df = load_raw()
ok(f"โหลด {RAW_PATH.name} สำเร็จ ({size_mb:,.1f} MB)")

show_kv(
    [
        ("จำนวนแถว (rows)", f"{df.shape[0]:,}"),
        ("จำนวนคอลัมน์ (fields)", df.shape[1]),
        ("คอลัมน์", ", ".join(df.columns)),
    ],
    title="ขนาดข้อมูล",
)
