"""Create a dated research snapshot from the official public sources."""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

from inflation_base_effects.data import refresh_snapshot


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--date",
        default=date.today().isoformat(),
        help="Snapshot directory name in YYYY-MM-DD format (default: today).",
    )
    args = parser.parse_args()
    target = Path("data") / "snapshots" / args.date
    created = refresh_snapshot(target)
    print(f"Created snapshot: {created}")


if __name__ == "__main__":
    main()
