# Model 1) เทรน Gradient Boosting ด้วยข้อมูลทั้งหมด
# โจทย์: ทำนายว่า "สินค้านี้จะขายดี (Top 10% ของเดือน) ในเดือนนี้ไหม" จากยอดขายเดือนก่อนๆ
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt
import pandas as pd

from common import (
    FEATURE_INFO,
    HOT_QUANTILE,
    MODEL_NAME,
    MODEL_REASON,
    TABLE_DIR,
    build_product_month,
    fig_confusion,
    fig_feature_importance,
    load_clean,
    predict_hot,
    save_fig,
    train_model,
)
from pretty import banner, ok, show_df, show_kv

banner(f"MODEL 1 — เทรน {MODEL_NAME}")

ok(MODEL_REASON)

clean = load_clean()
pm = build_product_month(clean)
show_df(
    pd.DataFrame({"feature (X)": list(FEATURE_INFO), "ความหมาย": list(FEATURE_INFO.values())}),
    title="Features ที่ใช้ (ข้อมูลย้อนหลังทั้งหมด ไม่ใช้ข้อมูลเดือนที่ทาย)",
)

metrics, eval_model, final_model, info = train_model(pm)
show_kv(
    [
        ("ข้อมูลหลัง clean", f"{len(clean):,} แถว (1 แถว = สินค้า 1 รายการในใบเสร็จ)"),
        ("รวมเป็น สินค้า × เดือน", f"{info['n_rows']:,} แถว (ใช้ข้อมูลทั้ง {len(clean):,} แถวมาสรุป)"),
        ("Target (y)", f"is_hot = 1 ถ้ายอดขายเดือนนั้นอยู่ Top {round((1 - HOT_QUANTILE) * 100)}% ของสินค้า"),
        ("สัดส่วนสินค้าขายดี", f"{info['hot_rate'] * 100:.1f}%"),
        ("Train (ทั้งหมด ไม่สุ่มตัด)", f"{info['n_train']:,} แถว  ({info['train_months']})"),
        ("Test", f"{info['n_test']:,} แถว  ({info['test_months']})"),
        ("โมเดลสุดท้าย", f"เทรนใหม่ด้วย Train + Test = {info['n_labeled']:,} แถว ({info['all_months']})"),
        ("ใช้ทำนาย", f"ธ.ค. 2011 ({info['n_forecast']:,} สินค้า)"),
    ],
    title="ข้อมูลที่ใช้เทรน",
)

show_df(pd.DataFrame([metrics]).round(3), title="ผลบน Test set")
ok(f"เทรนเสร็จใน {info['train_sec']:.1f} วินาที")

TABLE_DIR.mkdir(parents=True, exist_ok=True)
predict_hot(pm, eval_model, final_model).to_csv(TABLE_DIR / "hot_predictions.csv", index=False)
ok("บันทึก outputs/tables/hot_predictions.csv")

ok(f"บันทึก {save_fig(fig_confusion(info['cm']), '09_confusion_matrix.png')}")
ok(f"บันทึก {save_fig(fig_feature_importance(final_model), '09_feature_importance.png')}")
plt.show()
