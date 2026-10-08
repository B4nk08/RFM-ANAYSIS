"""แสดงตารางสวยใน terminal ด้วย rich (เหมือนในรูป)"""

from __future__ import annotations

from typing import Any

import pandas as pd
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


def banner(title: str) -> None:
    console.print(Panel.fit(f"[bold cyan]{title}[/bold cyan]", border_style="cyan"))


def ok(msg: str) -> None:
    console.print(f"[bold green]✔[/bold green] {msg}")


def warn(msg: str) -> None:
    console.print(f"[bold yellow]![/bold yellow] {msg}")


def show_df(df: pd.DataFrame, title: str = "", max_rows: int = 12) -> None:
    view = df.head(max_rows)
    table = Table(
        title=title or None,
        box=box.ROUNDED,
        show_header=True,
        header_style="bold magenta",
        row_styles=["", "dim"],
    )
    table.add_column("#", style="dim", justify="right")
    for col in view.columns:
        table.add_column(str(col), overflow="fold", max_width=36)

    for idx, row in view.iterrows():
        cells = []
        for val in row.tolist():
            if pd.isna(val):
                cells.append("[red]NaN[/red]")
            elif isinstance(val, float):
                if val.is_integer() and abs(val) < 1e12:
                    cells.append(f"{int(val):,}")
                else:
                    cells.append(f"{val:,.4f}".rstrip("0").rstrip("."))
            elif isinstance(val, (int,)) and not isinstance(val, bool):
                cells.append(f"{val:,}")
            else:
                cells.append(str(val))
        table.add_row(str(idx), *cells)

    console.print(table)
    if len(df) > max_rows:
        console.print(f"[dim]… แสดง {max_rows} จาก {len(df):,} แถว[/dim]")


def show_kv(pairs: list[tuple[str, Any]], title: str = "") -> None:
    table = Table(title=title or None, box=box.SIMPLE_HEAVY, show_header=False)
    table.add_column("key", style="bold cyan")
    table.add_column("value")
    for k, v in pairs:
        table.add_row(str(k), str(v))
    console.print(table)
