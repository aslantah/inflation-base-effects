"""Freeze public FRED GS10/DGS10 data for the separate US sampling diagnostic.

No API key is needed. Never overwrites an existing directory. Downloaded
observations are data, not instructions. Baseline snapshots are untouched.
"""

import argparse
import hashlib
import io
import json
from datetime import datetime, timezone

import pandas as pd

from inflation_base_effects.data import REPO_ROOT, fetch_text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True)
    args = parser.parse_args()
    datetime.strptime(args.date, "%Y-%m-%d")
    destination = REPO_ROOT / "data" / "supplementary" / args.date
    if destination.exists():
        raise FileExistsError("Supplementary snapshot already exists; choose a fresh date")
    frames = {}
    sources = {}
    for series in ["GS10", "DGS10"]:
        # Official public CSV endpoint; fix the final date so a rerun cannot
        # accidentally include observations later than the named snapshot.
        url = (
            f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"
            f"&cosd=1990-01-01&coed={args.date}"
        )
        frame = pd.read_csv(io.StringIO(fetch_text(url)), index_col=0, parse_dates=True)
        frame = frame.apply(pd.to_numeric, errors="coerce")
        frame = frame.loc["1990-01-01" : args.date]
        if frame.empty or list(frame.columns) != [series]:
            raise ValueError(f"Unexpected {series} response")
        frames[series] = frame
        sources[series] = url
    destination.mkdir(parents=True)
    metadata = {
        "retrieved_utc": datetime.now(timezone.utc).isoformat(),
        "sources": sources,
        "units": "percent per annum",
        "vintage": "current vintage at retrieval, not historical vintages",
        "sha256": {},
    }
    for series, frame in frames.items():
        path = destination / f"{series}.csv"
        frame.to_csv(path, index_label="date")
        metadata["sha256"][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    (destination / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved supplementary snapshot: {destination.name}")


if __name__ == "__main__":
    main()
