"""
Automated Ledger Reconciliation Engine
=======================================

Reconciles an external third-party transaction export against an internal
general ledger (GL) export. Designed for finance and engineering teams who
need auditable, repeatable close-process automation.

Data Sources
------------
- ``external_vendor_dump.csv``  : Messy vendor-side transaction export.
- ``internal_general_ledger.csv`` : Internal GL export (typically cleaner).

Output
------
- ``recon_variance_report.csv`` : Exception report containing only rows that
  require human review (missing on one side, or amount mismatches).

Usage
-----
    python reconcile_ledger.py

Requirements
------------
    pip install pandas
"""

from __future__ import annotations

import logging
import re
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

VENDOR_FILE = Path("external_vendor_dump.csv")
LEDGER_FILE = Path("internal_general_ledger.csv")
OUTPUT_FILE = Path("recon_variance_report.csv")

# Tolerance for floating-point amount comparisons (sub-penny).
AMOUNT_TOLERANCE = 0.005

# Canonical column names used internally after normalization.
COL_TXN_ID = "transaction_id"
COL_DATE = "transaction_date"
COL_AMOUNT = "amount"
COL_DESCRIPTION = "description"
COL_SOURCE = "source_system"

# Candidate header names found in real-world exports (case-insensitive).
TXN_ID_ALIASES = (
    "transaction_id",
    "txn_id",
    "trx_id",
    "trans_id",
    "transaction id",
    "id",
    "reference",
    "ref",
    "ref_no",
    "reference_number",
)

DATE_ALIASES = (
    "transaction_date",
    "txn_date",
    "date",
    "posting_date",
    "post_date",
    "value_date",
    "trans_date",
)

AMOUNT_ALIASES = (
    "amount",
    "transaction_amount",
    "txn_amount",
    "debit",
    "credit",
    "net_amount",
    "value",
)

DESCRIPTION_ALIASES = (
    "description",
    "memo",
    "narrative",
    "details",
    "transaction_description",
    "payee",
)

# Exception categories written to the variance report.
STATUS_LEDGER_ONLY = "LEDGER_ONLY"
STATUS_VENDOR_ONLY = "VENDOR_ONLY"
STATUS_AMOUNT_MISMATCH = "AMOUNT_MISMATCH"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data transformation helpers
# ---------------------------------------------------------------------------


def _normalize_header(name: str) -> str:
    """Collapse header text to a lowercase, underscore-delimited token."""
    cleaned = re.sub(r"[^\w\s]", "", str(name).strip().lower())
    return re.sub(r"\s+", "_", cleaned)


def _resolve_column(df: pd.DataFrame, aliases: tuple[str, ...]) -> Optional[str]:
    """
    Map a DataFrame column to the first alias that matches a normalized header.

    Returns the original column name from ``df``, or ``None`` if no match.
    """
    normalized = {_normalize_header(col): col for col in df.columns}
    for alias in aliases:
        if alias in normalized:
            return normalized[alias]
    return None


def clean_whitespace(series: pd.Series) -> pd.Series:
    """Strip leading/trailing whitespace and collapse internal runs of spaces."""
    return (
        series.astype(str)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
        .replace({"nan": pd.NA, "None": pd.NA, "": pd.NA})
    )


def parse_currency(series: pd.Series) -> pd.Series:
    """
    Convert currency-formatted strings to float.

    Handles common export artifacts: dollar signs, commas, parentheses for
    negatives (e.g. ``($1,250.00)``), and trailing debit/credit indicators.

    Examples
    --------
    ``"$1,250.00"``  -> 1250.0
    ``"(500.00)"``   -> -500.0
    ``"1,250.00 DR"`` -> 1250.0
    """
    def _to_float(value: object) -> float:
        if pd.isna(value):
            return float("nan")

        text = str(value).strip()
        if not text:
            return float("nan")

        # Parentheses denote negative amounts in accounting notation.
        is_negative = text.startswith("(") and text.endswith(")")
        if is_negative:
            text = text[1:-1]

        # Remove currency symbols, thousands separators, and CR/DR suffixes.
        text = re.sub(r"[$£€]", "", text)
        text = re.sub(r",", "", text)
        text = re.sub(r"\b(CR|DR)\b", "", text, flags=re.IGNORECASE).strip()

        try:
            amount = float(text)
        except ValueError:
            return float("nan")

        return -amount if is_negative else amount

    return series.apply(_to_float)


def parse_dates(series: pd.Series) -> pd.Series:
    """
    Standardize heterogeneous date strings to ``datetime64[ns]``.

    Uses pandas' inference engine with ``dayfirst=False`` (US-style default).
    Unparseable values become ``NaT`` and are logged at debug level.
    """
    cleaned = clean_whitespace(series)
    parsed = pd.to_datetime(cleaned, errors="coerce")
    invalid_count = parsed.isna().sum() - cleaned.isna().sum()
    if invalid_count > 0:
        logger.warning("%d date value(s) could not be parsed and were set to NaT.", invalid_count)
    return parsed


def standardize_dataframe(df: pd.DataFrame, source_label: str) -> pd.DataFrame:
    """
    Normalize a raw export into a consistent schema for reconciliation.

    Pipeline
    --------
    1. Resolve column aliases to canonical names.
    2. Clean text fields (whitespace normalization).
    3. Parse dates and currency amounts.
    4. Derive a composite match key (date + amount) as a fallback identifier.
    5. Tag each row with its originating system.

    Parameters
    ----------
    df : pd.DataFrame
        Raw export as loaded from CSV.
    source_label : str
        Human-readable source name (e.g. ``"VENDOR"`` or ``"LEDGER"``).

    Returns
    -------
    pd.DataFrame
        Standardized frame with canonical columns and derived match keys.

    Raises
    ------
    ValueError
        If required columns (date and amount) cannot be resolved.
    """
    working = df.copy()
    working.columns = [_normalize_header(c) for c in working.columns]

    txn_col = _resolve_column(working, TXN_ID_ALIASES)
    date_col = _resolve_column(working, DATE_ALIASES)
    amount_col = _resolve_column(working, AMOUNT_ALIASES)
    desc_col = _resolve_column(working, DESCRIPTION_ALIASES)

    if date_col is None or amount_col is None:
        raise ValueError(
            f"[{source_label}] Could not resolve required date/amount columns. "
            f"Available headers: {list(working.columns)}"
        )

    standardized = pd.DataFrame()
    standardized[COL_SOURCE] = source_label

    if txn_col:
        standardized[COL_TXN_ID] = clean_whitespace(working[txn_col]).str.upper()
    else:
        standardized[COL_TXN_ID] = pd.NA
        logger.warning("[%s] No transaction ID column detected; matching will use date + amount.", source_label)

    standardized[COL_DATE] = parse_dates(working[date_col])
    standardized[COL_AMOUNT] = parse_currency(working[amount_col])

    if desc_col:
        standardized[COL_DESCRIPTION] = clean_whitespace(working[desc_col])
    else:
        standardized[COL_DESCRIPTION] = pd.NA

    # Composite key: ISO date string + amount rounded to cents.
    # Used when transaction IDs are missing or unreliable across systems.
    standardized["match_key"] = (
        standardized[COL_DATE].dt.strftime("%Y-%m-%d")
        + "|"
        + standardized[COL_AMOUNT].round(2).astype(str)
    )

    # Prefer transaction ID for primary matching when present.
    standardized["primary_key"] = standardized[COL_TXN_ID].where(
        standardized[COL_TXN_ID].notna() & (standardized[COL_TXN_ID] != ""),
        standardized["match_key"],
    )

    return standardized


def amounts_equal(left: float, right: float, tolerance: float = AMOUNT_TOLERANCE) -> bool:
    """Return True if two amounts agree within the configured tolerance."""
    if pd.isna(left) or pd.isna(right):
        return False
    return abs(left - right) <= tolerance


# ---------------------------------------------------------------------------
# Reconciliation logic
# ---------------------------------------------------------------------------


def reconcile(vendor: pd.DataFrame, ledger: pd.DataFrame) -> pd.DataFrame:
    """
    Perform a full outer reconciliation between vendor and ledger DataFrames.

    Matching Strategy
    -----------------
    1. **Primary match** on ``transaction_id`` when both sides have a value.
    2. **Fallback match** on the composite ``match_key`` (date + amount) for
       rows without a reliable transaction ID.

    Exception Categories
    --------------------
    - ``LEDGER_ONLY``      : Present in GL, absent from vendor export.
    - ``VENDOR_ONLY``      : Present in vendor export, absent from GL.
    - ``AMOUNT_MISMATCH``  : Keys align but normalized amounts differ.

    Parameters
    ----------
    vendor : pd.DataFrame
        Standardized vendor export.
    ledger : pd.DataFrame
        Standardized GL export.

    Returns
    -------
    pd.DataFrame
        Variance report containing only exception rows, sorted for review.
    """
    merged = vendor.merge(
        ledger,
        on="primary_key",
        how="outer",
        suffixes=("_vendor", "_ledger"),
        indicator=True,
    )

    exceptions: list[pd.DataFrame] = []

    # --- Rows present on one side only ------------------------------------
    ledger_only = merged[merged["_merge"] == "right_only"].copy()
    if not ledger_only.empty:
        ledger_only["exception_status"] = STATUS_LEDGER_ONLY
        ledger_only["variance_amount"] = ledger_only[f"{COL_AMOUNT}_ledger"]
        exceptions.append(_format_exception_rows(ledger_only, "ledger"))

    vendor_only = merged[merged["_merge"] == "left_only"].copy()
    if not vendor_only.empty:
        vendor_only["exception_status"] = STATUS_VENDOR_ONLY
        vendor_only["variance_amount"] = vendor_only[f"{COL_AMOUNT}_vendor"]
        exceptions.append(_format_exception_rows(vendor_only, "vendor"))

    # --- Rows matched on key but with differing amounts -------------------
    matched = merged[merged["_merge"] == "both"].copy()
    if not matched.empty:
        amount_mismatch = matched[
            ~matched.apply(
                lambda row: amounts_equal(
                    row[f"{COL_AMOUNT}_vendor"],
                    row[f"{COL_AMOUNT}_ledger"],
                ),
                axis=1,
            )
        ].copy()

        if not amount_mismatch.empty:
            amount_mismatch["exception_status"] = STATUS_AMOUNT_MISMATCH
            amount_mismatch["variance_amount"] = (
                amount_mismatch[f"{COL_AMOUNT}_vendor"]
                - amount_mismatch[f"{COL_AMOUNT}_ledger"]
            )
            exceptions.append(_format_exception_rows(amount_mismatch, "both"))

    if not exceptions:
        logger.info("Reconciliation complete — no exceptions found.")
        return pd.DataFrame(
            columns=[
                "exception_status",
                "primary_key",
                "transaction_id",
                "transaction_date",
                "vendor_amount",
                "ledger_amount",
                "variance_amount",
                "description",
                "source_system",
            ]
        )

    report = pd.concat(exceptions, ignore_index=True)
    report = report.sort_values(
        by=["exception_status", "transaction_date"],
        ascending=[True, True],
        na_position="last",
    ).reset_index(drop=True)

    return report


def _format_exception_rows(df: pd.DataFrame, match_side: str) -> pd.DataFrame:
    """
    Project merged reconciliation rows into the final variance report schema.

    Parameters
    ----------
    df : pd.DataFrame
        Subset of the outer-joined frame for a single exception type.
    match_side : str
        ``"vendor"``, ``"ledger"``, or ``"both"`` — controls which columns populate.
    """
    formatted = pd.DataFrame()
    formatted["exception_status"] = df["exception_status"]
    formatted["primary_key"] = df["primary_key"]

    if match_side in ("vendor", "both"):
        formatted["transaction_id"] = df.get(f"{COL_TXN_ID}_vendor", pd.NA)
        formatted["transaction_date"] = df.get(f"{COL_DATE}_vendor", pd.NA)
        formatted["vendor_amount"] = df.get(f"{COL_AMOUNT}_vendor", pd.NA)
        formatted["description"] = df.get(f"{COL_DESCRIPTION}_vendor", pd.NA)
    else:
        formatted["transaction_id"] = pd.NA
        formatted["transaction_date"] = pd.NA
        formatted["vendor_amount"] = pd.NA
        formatted["description"] = pd.NA

    if match_side in ("ledger", "both"):
        if match_side == "ledger":
            formatted["transaction_id"] = df.get(f"{COL_TXN_ID}_ledger", pd.NA)
            formatted["transaction_date"] = df.get(f"{COL_DATE}_ledger", pd.NA)
            formatted["description"] = df.get(f"{COL_DESCRIPTION}_ledger", pd.NA)
        formatted["ledger_amount"] = df.get(f"{COL_AMOUNT}_ledger", pd.NA)
    else:
        formatted["ledger_amount"] = pd.NA

    formatted["variance_amount"] = df["variance_amount"]

    if match_side == "ledger":
        formatted["source_system"] = df.get(f"{COL_SOURCE}_ledger", "LEDGER")
    elif match_side == "vendor":
        formatted["source_system"] = df.get(f"{COL_SOURCE}_vendor", "VENDOR")
    else:
        formatted["source_system"] = "BOTH"

    return formatted[
        [
            "exception_status",
            "primary_key",
            "transaction_id",
            "transaction_date",
            "vendor_amount",
            "ledger_amount",
            "variance_amount",
            "description",
            "source_system",
        ]
    ]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def load_csv(path: Path) -> pd.DataFrame:
    """Load a CSV file, stripping BOM characters from headers if present."""
    if not path.exists():
        raise FileNotFoundError(f"Required input file not found: {path}")
    logger.info("Loading %s", path)
    return pd.read_csv(path, dtype=str, encoding="utf-8-sig")


def print_summary(report: pd.DataFrame) -> None:
    """Log a concise reconciliation summary for the operations team."""
    if report.empty:
        logger.info("Summary: 0 exceptions — books are in balance.")
        return

    counts = report["exception_status"].value_counts()
    logger.info("--- Reconciliation Summary ---")
    for status, count in counts.items():
        logger.info("  %-20s %d", status, count)
    logger.info("  %-20s %d", "TOTAL EXCEPTIONS", len(report))


def main() -> int:
    """
    Execute the end-to-end reconciliation pipeline.

    Returns
    -------
    int
        Process exit code (0 = success, 1 = failure).
    """
    try:
        # Stage 1: Ingest raw exports ----------------------------------------
        vendor_raw = load_csv(VENDOR_FILE)
        ledger_raw = load_csv(LEDGER_FILE)
        logger.info("Vendor rows: %d | Ledger rows: %d", len(vendor_raw), len(ledger_raw))

        # Stage 2: Normalize and clean ---------------------------------------
        vendor_clean = standardize_dataframe(vendor_raw, source_label="VENDOR")
        ledger_clean = standardize_dataframe(ledger_raw, source_label="LEDGER")

        # Stage 3: Reconcile and classify exceptions -------------------------
        variance_report = reconcile(vendor_clean, ledger_clean)

        # Stage 4: Persist exception report ----------------------------------
        variance_report.to_csv(OUTPUT_FILE, index=False, float_format="%.2f")
        logger.info("Variance report written to %s", OUTPUT_FILE)

        print_summary(variance_report)
        return 0

    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1
    except ValueError as exc:
        logger.error("Data validation error: %s", exc)
        return 1
    except Exception:
        logger.exception("Unexpected error during reconciliation.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
