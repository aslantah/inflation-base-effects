"""Verified, reproducible data access for the base-effects study."""

from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import re
import urllib.error
import urllib.parse
import urllib.request
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
DataSource = Literal["snapshot", "live"]
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SNAPSHOT_DIR = REPO_ROOT / "data" / "snapshots" / "2026-09-28"


class DataAccessError(RuntimeError):
    """Raised when an official data source cannot be downloaded or parsed."""


def _load_env_file() -> None:
    """Load a local ``.env`` without overriding an existing environment value."""
    try:
        from dotenv import load_dotenv

        load_dotenv(override=False)
        return
    except ImportError:
        pass

    search = Path(__file__).resolve().parent
    for parent in [search, *search.parents[:5]]:
        candidate = parent / ".env"
        if not candidate.is_file():
            continue
        for line in candidate.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip("\"'"))
        return


_load_env_file()


def _require_api_key() -> str:
    key = os.environ.get("FRED_API_KEY", "")
    if not key:
        raise DataAccessError(
            "FRED_API_KEY is required for a live refresh. Copy .env.example to .env "
            "and add a free key from https://fred.stlouisfed.org/docs/api/api_key.html."
        )
    return key


def _urlopen(request: urllib.request.Request, timeout: int = 30):
    """Open an HTTPS request with Python's verified default SSL context."""
    try:
        return urllib.request.urlopen(request, timeout=timeout)
    except urllib.error.URLError as exc:
        reason = getattr(exc, "reason", exc)
        if "CERTIFICATE_VERIFY_FAILED" in str(reason):
            raise DataAccessError(
                "TLS certificate verification failed. Configure SSL_CERT_FILE with the "
                "required CA bundle; certificate checks are intentionally not disabled."
            ) from exc
        raise


def fetch_fred(
    series_id: str,
    start: str = "1990-01-01",
    end: str | None = None,
    retries: int = 3,
    backoff: float = 2.0,
) -> pd.Series:
    """Fetch one FRED series using verified HTTPS and bounded retries."""
    import time

    params = {
        "series_id": series_id,
        "api_key": _require_api_key(),
        "file_type": "json",
        "observation_start": start,
    }
    if end:
        params["observation_end"] = end
    url = "https://api.stlouisfed.org/fred/series/observations?" + urllib.parse.urlencode(params)
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(
                url, headers={"User-Agent": "inflation-base-effects/0.1"}
            )
            with _urlopen(request) as response:
                observations = json.loads(response.read().decode("utf-8"))["observations"]
            values = [
                float(item["value"]) if item["value"] != "." else np.nan for item in observations
            ]
            dates = pd.to_datetime([item["date"] for item in observations])
            return pd.Series(values, index=dates, name=series_id).dropna()
        except urllib.error.HTTPError as exc:
            last_exc = exc
            if exc.code not in (429, 502, 503) or attempt == retries - 1:
                raise DataAccessError(
                    f"FRED download failed for {series_id}: HTTP {exc.code}"
                ) from exc
        except (urllib.error.URLError, KeyError, ValueError, json.JSONDecodeError) as exc:
            last_exc = exc
            if attempt == retries - 1:
                raise DataAccessError(f"FRED download failed for {series_id}: {exc}") from exc
        time.sleep(backoff**attempt)
    raise DataAccessError(f"FRED download failed for {series_id}: {last_exc}")


def fetch_text(url: str, retries: int = 3, backoff: float = 2.0) -> str:
    """Download UTF-8 text from an official source."""
    import time

    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(
                url, headers={"User-Agent": "inflation-base-effects/0.1"}
            )
            with _urlopen(request) as response:
                return response.read().decode("utf-8-sig")
        except (urllib.error.HTTPError, urllib.error.URLError, UnicodeDecodeError) as exc:
            last_exc = exc
            if attempt == retries - 1:
                break
            time.sleep(backoff**attempt)
    raise DataAccessError(f"Text download failed for {url}: {last_exc}")


def fetch_remote_excel(url: str, retries: int = 3, backoff: float = 2.0) -> pd.DataFrame:
    """Download and validate an XLSX workbook from an official source."""
    import time
    import zipfile

    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(
                url, headers={"User-Agent": "inflation-base-effects/0.1"}
            )
            with _urlopen(request) as response:
                raw = response.read()
            if raw[:2] != b"PK":
                raise ValueError("response is not an XLSX ZIP archive")
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Cannot parse header or footer")
                return pd.read_excel(io.BytesIO(raw), engine="openpyxl")
        except (
            urllib.error.HTTPError,
            urllib.error.URLError,
            ValueError,
            zipfile.BadZipFile,
        ) as exc:
            last_exc = exc
            if attempt == retries - 1:
                break
            time.sleep(backoff**attempt)
    raise DataAccessError(f"Excel download failed for {url}: {last_exc}")


CPI_SERIES: dict[str, str] = {
    "B_EMU_BD": "DEUCPALTT01IXNBM",
    "B_UKI_BD": "GBRCPALTT01IXNBM",
    "B_CAN_BD": "CANCPALTT01IXNBM",
    "B_USA_BD": "CPIAUCNS",
}
YIELD_SERIES: dict[str, str] = {
    "B_USA_BD": "GS10",
    "B_EMU_BD": "IRLTLT01DEM156N",
    "B_UKI_BD": "IRLTLT01GBM156N",
    "B_CAN_BD": "IRLTLT01CAM156N",
}
SPF_URL = (
    "https://www.philadelphiafed.org/-/media/frbp/assets/surveys-and-data/"
    "survey-of-professional-forecasters/data-files/files/Median_CPI_Level.xlsx"
)
SPF_RELEASE_DATES_URL = (
    "https://www.philadelphiafed.org/-/media/frbp/assets/surveys-and-data/"
    "survey-of-professional-forecasters/spf-release-dates.txt"
)
UNIVERSE = ["B_EMU_BD", "B_UKI_BD", "B_CAN_BD", "B_USA_BD"]


def _snapshot_path(snapshot_dir: str | Path | None, filename: str) -> Path:
    base = Path(snapshot_dir) if snapshot_dir is not None else DEFAULT_SNAPSHOT_DIR
    path = base / filename
    if not path.is_file():
        raise DataAccessError(
            f"Snapshot file not found: {path}. Run "
            "`python scripts/refresh_data.py --date YYYY-MM-DD`."
        )
    return path


def _read_snapshot(snapshot_dir: str | Path | None, filename: str) -> pd.DataFrame:
    frame = pd.read_csv(
        _snapshot_path(snapshot_dir, filename), index_col="date", parse_dates=["date"]
    )
    frame.index.name = None
    return frame.sort_index()


def _live_cpi() -> pd.DataFrame:
    frame = pd.DataFrame({name: fetch_fred(series) for name, series in CPI_SERIES.items()})
    frame.index = frame.index.to_period("M").to_timestamp()
    return frame.sort_index().dropna(how="all")


def _live_yields() -> pd.DataFrame:
    parts = {}
    for name, series in YIELD_SERIES.items():
        values = fetch_fred(series)
        values.index = values.index.to_period("M").to_timestamp()
        parts[name] = values[~values.index.duplicated(keep="last")]
    return pd.DataFrame(parts).sort_index().dropna(how="all")


def _live_michigan() -> pd.DataFrame:
    values = fetch_fred("MICH")
    values.index = values.index.to_period("M").to_timestamp()
    return values.sort_index().diff().dropna().to_frame("B_USA_BD")


def parse_spf_release_dates(text: str) -> pd.DataFrame:
    """Parse the Philadelphia Fed's official deadline/release-date file."""
    records: list[dict[str, object]] = []
    current_year: int | None = None
    pattern = re.compile(
        r"^(?:(?P<year>\d{4})\s+)?Q(?P<quarter>[1-4])\s+"
        r"(?P<deadline>\d{1,2}/\d{1,2}/\d{2})\*{0,3}\s+"
        r"(?P<release>\d{1,2}/\d{1,2}/\d{2})\*{0,3}$"
    )
    for raw_line in text.splitlines():
        match = pattern.match(" ".join(raw_line.strip().split()))
        if not match:
            continue
        if match.group("year"):
            current_year = int(match.group("year"))
        if current_year is None:
            continue
        quarter = int(match.group("quarter"))
        records.append(
            {
                "survey": f"{current_year}Q{quarter}",
                "deadline_date": pd.to_datetime(match.group("deadline"), format="%m/%d/%y"),
                "release_date": pd.to_datetime(match.group("release"), format="%m/%d/%y"),
            }
        )
    if not records:
        raise DataAccessError("No SPF release dates could be parsed from the official text file.")
    return pd.DataFrame(records).set_index("survey").sort_index()


def _live_spf_with_dates() -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = fetch_remote_excel(SPF_URL)
    for column in raw.columns:
        raw[column] = pd.to_numeric(raw[column], errors="coerce")
    raw = raw.dropna(subset=["YEAR", "QUARTER"]).copy()
    raw["survey"] = (
        raw["YEAR"].astype(int).astype(str) + "Q" + raw["QUARTER"].astype(int).astype(str)
    )
    raw = raw.sort_values(["YEAR", "QUARTER"])
    raw["revision"] = raw["CPI5"] - raw["CPI6"].shift(1)
    dates = parse_spf_release_dates(fetch_text(SPF_RELEASE_DATES_URL))
    merged = raw.set_index("survey")[["revision"]].join(dates, how="inner").dropna()
    month_index = merged["deadline_date"].dt.to_period("M").dt.to_timestamp()
    revisions = pd.DataFrame({"B_USA_BD": merged["revision"].to_numpy()}, index=month_index)
    revisions.index.name = None
    return revisions.sort_index(), dates


def load_cpi(
    source: DataSource = "snapshot", snapshot_dir: str | Path | None = None
) -> pd.DataFrame:
    """Load monthly NSA CPI price levels."""
    return _read_snapshot(snapshot_dir, "cpi.csv") if source == "snapshot" else _live_cpi()


def load_yields(
    source: DataSource = "snapshot", snapshot_dir: str | Path | None = None
) -> pd.DataFrame:
    """Load comparable monthly-average 10-year government bond yields."""
    return _read_snapshot(snapshot_dir, "yields.csv") if source == "snapshot" else _live_yields()


def load_michigan(
    source: DataSource = "snapshot", snapshot_dir: str | Path | None = None
) -> pd.DataFrame:
    """Load monthly changes in Michigan one-year inflation expectations."""
    return (
        _read_snapshot(snapshot_dir, "michigan.csv") if source == "snapshot" else _live_michigan()
    )


def load_spf(
    source: DataSource = "snapshot", snapshot_dir: str | Path | None = None
) -> pd.DataFrame:
    """Load same-target SPF revisions indexed to the official deadline month."""
    if source == "snapshot":
        return _read_snapshot(snapshot_dir, "spf.csv")
    revisions, _ = _live_spf_with_dates()
    return revisions


def prepare_inflation_data(
    headline_nsa: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute annualized MoM, YoY, and full-sample seasonally adjusted MoM CPI."""
    import statsmodels.api as sm

    cpi_mom = (headline_nsa / headline_nsa.shift(1)) ** 12 - 1
    cpi_yoy = (headline_nsa / headline_nsa.shift(12) - 1).dropna(how="all")
    parts = {}
    for column in cpi_mom.columns:
        series = cpi_mom[column].dropna()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            decomposition = sm.tsa.seasonal_decompose(series, period=12, two_sided=False)
        parts[column] = series - decomposition.seasonal
    return cpi_mom, cpi_yoy, pd.DataFrame(parts).dropna()


def compute_base_effect(headline_nsa: pd.DataFrame) -> pd.DataFrame:
    """Return the NSA monthly log-inflation print rolling out of the YoY window."""
    return np.log(headline_nsa).diff().shift(12).dropna(how="all")


def _write_frame(frame: pd.DataFrame, path: Path) -> None:
    output = frame.copy()
    output.index.name = "date"
    output.to_csv(path, date_format="%Y-%m-%d", float_format="%.12g")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_snapshot(snapshot_dir: str | Path | None = None) -> None:
    """Validate required files, declared checksums, and core snapshot schemas."""
    base = Path(snapshot_dir) if snapshot_dir is not None else DEFAULT_SNAPSHOT_DIR
    checksum_path = base / "checksums.json"
    manifest_path = base / "manifest.json"
    if not checksum_path.is_file() or not manifest_path.is_file():
        raise DataAccessError(f"Snapshot metadata is incomplete in {base}")
    checksums = json.loads(checksum_path.read_text(encoding="utf-8"))
    for filename, expected in checksums.items():
        path = base / filename
        if not path.is_file() or _sha256(path) != expected:
            raise DataAccessError(f"Snapshot checksum mismatch: {path}")
    expected_columns = {
        "cpi.csv": set(UNIVERSE),
        "yields.csv": set(UNIVERSE),
        "michigan.csv": {"B_USA_BD"},
        "spf.csv": {"B_USA_BD"},
    }
    for filename, columns in expected_columns.items():
        frame = _read_snapshot(base, filename)
        if set(frame.columns) != columns or frame.empty:
            raise DataAccessError(f"Unexpected schema in snapshot file: {base / filename}")


def refresh_snapshot(output_dir: str | Path) -> Path:
    """Create a new, non-overwriting snapshot from official live sources."""
    target = Path(output_dir)
    if target.exists() and any(target.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty snapshot directory: {target}")
    target.mkdir(parents=True, exist_ok=True)
    frames = {
        "cpi.csv": _live_cpi(),
        "yields.csv": _live_yields(),
        "michigan.csv": _live_michigan(),
    }
    spf, release_dates = _live_spf_with_dates()
    frames["spf.csv"] = spf
    for filename, frame in frames.items():
        _write_frame(frame, target / filename)
    release_dates.to_csv(target / "spf_release_dates.csv", date_format="%Y-%m-%d")
    files = sorted(path.name for path in target.iterdir() if path.is_file())
    manifest = {
        "snapshot_date": target.name,
        "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
        "sources": {
            "fred_cpi": CPI_SERIES,
            "fred_yields": YIELD_SERIES,
            "fred_michigan": "MICH",
            "philadelphia_fed_spf": SPF_URL,
            "philadelphia_fed_spf_dates": SPF_RELEASE_DATES_URL,
        },
        "coverage": {
            filename: {
                "first": frame.dropna(how="all").index.min().strftime("%Y-%m-%d"),
                "last": frame.dropna(how="all").index.max().strftime("%Y-%m-%d"),
                "rows": len(frame),
            }
            for filename, frame in frames.items()
        },
    }
    manifest_path = target / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    checksums = {filename: _sha256(target / filename) for filename in files + ["manifest.json"]}
    (target / "checksums.json").write_text(
        json.dumps(checksums, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return target
