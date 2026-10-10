# Future Work — แนวทางต่อยอด
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

from common import FUTURE_WORK
from pretty import banner, show_df

banner("FUTURE WORK — ต่อยอดได้อย่างไร")

show_df(
    pd.DataFrame(FUTURE_WORK, columns=["หัวข้อ", "เทคนิค", "ประโยชน์"]),
    title="1) แนวทางต่อยอด",
)
