# Preprocessing 4) เช็คค่า missing values + outliers
# กราฟ: bar chart ของ missing และ boxplot ของ outlier
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt

from common import fig_boxplot, fig_missing, load_raw, missing_table, outlier_table, save_fig
from pretty import banner, ok, show_df

banner("PREPROCESSING 4 — เช็ค Missing Values และ Outliers")

df = load_raw()

show_df(missing_table(df), title="Missing values ต่อคอลัมน์")
show_df(outlier_table(df), title="Outliers (วิธี IQR) และค่าที่ ≤ 0")
ok(f"แถวที่ซ้ำกันทั้งแถว: {df.duplicated().sum():,}")

ok(f"บันทึก {save_fig(fig_missing(df), '04_missing_bar.png')}")
ok(f"บันทึก {save_fig(fig_boxplot(df), '04_outlier_boxplot.png')}")
plt.show()
