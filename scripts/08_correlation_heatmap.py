# Data Visualization 2) Correlation heatmap ของ Quantity, Price และ TotalPrice
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt

from common import fig_heatmap, load_clean, save_fig
from pretty import banner, ok, show_df

banner("VISUALIZATION 2 — Correlation Heatmap")

df = load_clean()
fig, corr = fig_heatmap(df)

show_df(corr.round(3).reset_index().rename(columns={"index": "feature"}), title="Correlation matrix")
ok("Quantity ↔ TotalPrice สัมพันธ์สูงสุด → จำนวนชิ้นคือตัวขับยอดขาย")
ok("Quantity ↔ Price ใกล้ 0 → ราคาไม่ได้กำหนดจำนวนซื้อแบบเส้นตรง")

ok(f"บันทึก {save_fig(fig, '08_correlation_heatmap.png')}")
plt.show()
