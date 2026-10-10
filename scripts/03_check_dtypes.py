# Preprocessing 3) เช็คชนิดข้อมูล
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import dtype_table, load_raw
from pretty import banner, show_df, warn

banner("PREPROCESSING 3 — เช็คชนิดข้อมูล (dtypes)")

df = load_raw()
show_df(dtype_table(df), title="ชนิดข้อมูลแต่ละคอลัมน์")

warn("InvoiceDate เป็นข้อความ (object) → ต้องแปลงเป็น datetime ตอน clean")
warn("Customer ID เป็น float เพราะมีค่าว่าง (NaN) → หลังตัดค่าว่างแปลงเป็น int")
