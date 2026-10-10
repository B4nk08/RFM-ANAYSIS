# Preprocessing 5) ใช้วิธีอะไรในการจัดการ missing value
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import CLEAN_PATH, MISSING_DECISIONS, clean_data, impute_comparison, load_raw
from pretty import banner, ok, show_df

banner("PREPROCESSING 5 — จัดการ Missing Value")

raw = load_raw()

show_df(impute_comparison(raw), title="ทดลองเติม Customer ID ด้วย Min / Max / Mean / Median / Mode")
show_df(MISSING_DECISIONS, title="วิธีที่เลือกใช้จริง")

clean, steps = clean_data(raw)
show_df(steps, title="ผลของแต่ละขั้นตอน")

clean.to_csv(CLEAN_PATH, index=False)
ok(f"บันทึกข้อมูลที่ clean แล้ว → {CLEAN_PATH.relative_to(CLEAN_PATH.parents[1])}")
