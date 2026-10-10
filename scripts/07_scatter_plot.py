# Data Visualization 1) Scatter plot ของ Quantity กับ Price (สเกลปกติ และ log)
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt

from common import fig_scatter, load_clean, save_fig
from pretty import banner, ok, show_kv

banner("VISUALIZATION 1 — Scatter: Quantity vs Price")

df = load_clean()
show_kv(
    [
        ("Correlation (Quantity, Price)", f"{df['Quantity'].corr(df['Price']):.3f}"),
        ("Median Quantity", df["Quantity"].median()),
        ("Median Price (£)", df["Price"].median()),
    ],
    title="ค่าที่ควรรู้",
)
ok("สเกลปกติ: จุดกองมุมซ้ายล่าง เพราะมีค่าสุดโต่ง")
ok("สเกล log: เห็นชัดว่าของราคาสูงมักซื้อจำนวนน้อย")

ok(f"บันทึก {save_fig(fig_scatter(df), '07_scatter_quantity_price.png')}")
plt.show()
