python3 scripts/1_load_data.py
python3 scripts/2_explore_data.py
python3 scripts/3_statistics.py
python3 scripts/4_handle_missing.py
python3 scripts/5_scatter_plots.py
python3 scripts/6_correlation_heatmap.py

# (ทางเลือก) รายงาน HTML
python3 scripts/7_make_report.py

# ----- หาคู่สินค้า + จัดโปรซื้อคู่ -----
python3 scripts/8_market_basket.py      # หาคู่สินค้า (Support/Confidence/Lift)
python3 scripts/9_pair_by_month.py      # คู่ไหนฮิตเดือนไหน
python3 scripts/10_promo_and_model.py   # แนะนำโปร + โมเดลเบาๆ

# ----- แอปเลือกสินค้า / เดือน (เปิดในเบราว์เซอร์) -----
# pip3 install streamlit --break-system-packages
streamlit run scripts/app_promo.py
