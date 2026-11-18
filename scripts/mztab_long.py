"""Write an mzTab file as long tables: one Parquet file per section and sample axis.

Each table holds a row per feature and sample, with that sample's metadata joined in, so a
reader can see what a parse rule would need to declare. ``MTD.tsv`` keeps every metadata line,
``COM.tsv`` the comment lines, ``samples.<axis>.tsv`` the sample tables. Every value stays text.
"""

from __future__ import annotations

import sys
from pathlib import Path

from cyclopts import App

from apb2.parserV2.joins.mztab import MzTabFormatError, write_long_tables

app = App(name="mztab-long", help=__doc__)


@app.default
def main(source: Path, out_dir: Path, /) -> None:
    """Write the long tables of ``source`` into ``out_dir``."""
    try:
        result = write_long_tables(source, out_dir)
    except MzTabFormatError as error:
        print(error, file=sys.stderr)
        sys.exit(1)
    print(f"{source}: mzTab {result.version or 'without a version line'}")
    if result.comments:
        print(f"  {result.comments} COM lines in COM.tsv")
    for table in result.tables:
        print(f"  {table.path.name}: {table.rows} rows, {table.columns} columns")


if __name__ == "__main__":
    app()
