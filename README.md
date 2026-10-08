# RFM Analysis — Online Retail II

โปรเจกต์ Data Mining: สำรวจข้อมูล + หาคู่สินค้าสำหรับโปรซื้อคู่

## โครงสร้าง

```
rfm-analysis/
├── data/
│   ├── online_retail_II.csv
│   └── cleaned_data.csv          # จากข้อ 4
├── scripts/
│   ├── 1_load_data.py … 7_make_report.py
│   ├── 8_market_basket.py        # หาคู่สินค้า
│   ├── 9_pair_by_month.py        # คู่ไหนฮิตเดือนไหน
│   └── 10_promo_and_model.py     # แนะนำโปร + โมเดลเบาๆ
├── outputs/
│   ├── figures/
│   ├── tables/                   # product_pairs, promo_suggestions
│   └── report.html
├── requirements.txt
├── Run.md
└── README.md
```

## ติดตั้ง

```bash
pip3 install -r requirements.txt --break-system-packages
```

## รัน (ดู Run.md)

```bash
# clean ข้อมูลก่อน (ถ้ายังไม่มี cleaned_data.csv)
python3 scripts/4_handle_missing.py

# หาคู่สินค้า + เดือน + โปร
python3 scripts/8_market_basket.py
python3 scripts/9_pair_by_month.py
python3 scripts/10_promo_and_model.py
```

ผลลัพธ์สำคัญ:
- `outputs/tables/product_pairs.csv` — คู่สินค้า + Lift
- `outputs/tables/pairs_best_month.csv` — เดือนที่ฮิต
- `outputs/tables/promo_suggestions.csv` — แนะนำโปร
- `outputs/figures/8_*.png`, `9_*.png`, `10_*.png`

## ความหมายตัวเลข

| ค่า | ความหมาย |
|---|---|
| Support | คู่นี้โผล่ในกี่ % ของใบเสร็จ |
| Confidence | ถ้าซื้อ A แล้วซื้อ B ด้วยกี่ % |
| Lift > 1 | ซื้อคู่กันมากกว่าแบบสุ่ม → เหมาะจัดโปร |

Online Retail II = บันทึกการขายออนไลน์ร้านของขวัญ UK ปี 2009–2011
