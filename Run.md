# วิธีรัน (รันจากโฟลเดอร์ rfm-analysis)

# ----- Preprocessing -----
python3 scripts/01_prepare_data.py       # 1) data ชื่ออะไร มาจากไหน
python3 scripts/02_show_data.py          # 2) 10 แถวแรก + จำนวนข้อมูล + แต่ละ field
python3 scripts/03_check_dtypes.py       # 3) ชนิดข้อมูล
python3 scripts/04_check_missing.py      # 4) missing + outliers (bar chart, boxplot)
python3 scripts/05_handle_missing.py     # 5) วิธีจัดการ missing → data/cleaned_data.csv
python3 scripts/06_data_summary.py       # 6) สรุปข้อมูลที่เหลือ

# ----- Data Visualization -----
python3 scripts/07_scatter_plot.py       # 1) Scatter Quantity vs Price (ปกติ + log)
python3 scripts/08_correlation_heatmap.py  # 2) Correlation heatmap

# ----- Model -----
python3 scripts/09_train_model.py        # 1) เทรน Gradient Boosting ด้วยข้อมูลทั้งหมด
python3 scripts/10_monthly_promo.py      # 2) โปรสินค้าขายดี (เดือนล่าสุด)
python3 scripts/10_monthly_promo.py 2011-11   #    เลือกเดือนเอง

# ----- Future Work -----
python3 scripts/11_future_work.py

# ----- หน้าเว็บรวมทุกขั้นตอน (หน้าเดียว) -----
streamlit run app.py
