# Model 2) เอาผลทำนายมาจัดโปรลดราคาสินค้าที่จะขายดีในแต่ละเดือน
# รัน:  python3 scripts/10_monthly_promo.py            (เดือนล่าสุด)
#       python3 scripts/10_monthly_promo.py 2011-11    (เลือกเดือน)
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

from common import FORECAST_MONTH, TABLE_DIR, month_label, promo_all_months, promo_for_month
from pretty import banner, ok, show_df, warn

banner("MODEL 2 — จัดโปรสินค้าขายดีรายเดือน")

pred_path = TABLE_DIR / "hot_predictions.csv"
if not pred_path.exists():
    warn("ยังไม่มีผลทำนาย → รัน python3 scripts/09_train_model.py ก่อน")
    sys.exit(1)

pred = pd.read_csv(pred_path, dtype={"StockCode": str})
months = sorted(pred["YearMonth"].unique())
ym = sys.argv[1] if len(sys.argv) > 1 else FORECAST_MONTH
if ym not in months:
    warn(f"ไม่มีเดือน {ym} — เลือกได้: {', '.join(months)}")
    sys.exit(1)

period = pred.loc[pred["YearMonth"] == ym, "period"].iloc[0]
ok(f"เดือน {month_label(ym)} ({ym}) · ชุดข้อมูล: {period}")
show_df(promo_for_month(pred, ym, show_actual=False), title=f"สินค้าที่ควรจัดโปร — {month_label(ym)}", max_rows=10)
ok("ส่วนลด: อันดับ 1–3 = 15% · 4–6 = 10% · 7–10 = 5%")

promo_all_months(pred).to_csv(TABLE_DIR / "monthly_promo.csv", index=False)
ok("บันทึกโปรทุกเดือน → outputs/tables/monthly_promo.csv")
