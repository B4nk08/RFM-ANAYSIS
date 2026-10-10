# Online Retail II — Data Mining Project

Preprocessing → Data Visualization → เทรน Gradient Boosting → จัดโปรสินค้าขายดีรายเดือน

## โครงสร้าง

```
rfm-analysis/
├── app.py                       # หน้าเว็บนำเสนอ (หน้าเดียว)
├── data/
│   ├── online_retail_II.csv     # ข้อมูลดิบ
│   └── cleaned_data.csv         # ได้จากข้อ 05
├── scripts/
│   ├── common.py                # ฟังก์ชันที่ใช้ร่วมกัน (terminal + เว็บ)
│   ├── pretty.py                # ตารางสวยใน terminal (rich)
│   ├── 01_prepare_data.py … 06_data_summary.py      # Preprocessing
│   ├── 07_scatter_plot.py, 08_correlation_heatmap.py # Visualization
│   ├── 09_train_model.py, 10_monthly_promo.py        # Model
│   └── 11_future_work.py
├── outputs/
│   ├── figures/                 # กราฟ .png
│   └── tables/                  # hot_predictions, monthly_promo
├── requirements.txt
└── Run.md                       # คำสั่งรันทีละข้อ
```

## ติดตั้ง

```bash
pip3 install -r requirements.txt --break-system-packages
```

## สรุปวิธีการ

| ส่วน | สิ่งที่ทำ |
|---|---|
| Missing | Customer ID → ตัดทิ้ง (เติมค่าแล้วเกิดลูกค้าปลอม) · Description → "Unknown" · Quantity/Price ≤ 0 → ตัดทิ้ง |
| ผลลัพธ์ | 1,067,371 → 805,549 แถว (เหลือ 75.5%) |
| โจทย์ Model | สินค้าจะขายดี (Top 10% ของเดือน) ไหม จากยอดขาย 1–3 เดือนก่อน |
| Model | Gradient Boosting (จับความสัมพันธ์ไม่เป็นเส้นตรง, ทนค่าสุดโต่ง, ไม่ต้อง scale, อธิบายด้วย feature importance ได้) |
| ข้อมูลเทรน | 805,549 แถว → รวมเป็นสินค้า × เดือน 59,988 แถว · Train 41,044 (2010-03 → 2011-05) · Test 16,063 (2011-06 → 2011-11) |
| โมเดลสุดท้าย | เทรนใหม่ด้วย Train + Test 57,107 แถว เพื่อพยากรณ์ ธ.ค. 2011 |
| โปร | Top 10 โอกาสขายดีของเดือน · อันดับ 1–3 ลด 15%, 4–6 ลด 10%, 7–10 ลด 5% |
