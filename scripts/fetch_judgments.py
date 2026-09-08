"""Fetch Indian court judgment PDFs from the AWS Open Data mirrors of eCourts.

Two public buckets (no AWS account, no credentials, CC-BY-4.0):

    indian-supreme-court-judgments     reported SCR judgments, 1950-present
    indian-high-court-judgments        25 High Courts, 1950-present (~17.8M docs)

We talk to them over plain HTTPS (public ListBucket + GetObject), so this needs
no boto3 and no awscli -- just ``requests``, ``pandas``, ``pyarrow``.

Each run writes PDFs under ``data/judgments/<court>/<year>/`` plus a
``manifest.csv`` carrying the real legal metadata (citation, bench, judge,
decision date, disposal). Downstream ingestion attaches those columns to every
chunk, which is what makes citation-accurate answers and the GraphRAG node
schema possible at all.

Usage
-----
    python scripts/fetch_judgments.py courts
    python scripts/fetch_judgments.py sc --years 2023-2025 --limit 300
    python scripts/fetch_judgments.py hc --years 2024 --courts 27_1,1_12 --limit 200
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from posixpath import basename

import pandas as pd
import requests

S3_NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"
SC_BUCKET = "indian-supreme-court-judgments"
HC_BUCKET = "indian-high-court-judgments"
OUT_ROOT = Path("data/judgments")

# Metadata columns worth keeping. Anything not present in a given bucket's
# parquet is simply left blank -- the two schemas overlap but are not identical.
MANIFEST_COLUMNS = [
    "pdf", "court", "year", "title", "citation", "cnr", "decision_date",
    "judge", "disposal_nature", "case_type", "bench_name", "petitioner",
    "respondent",
]


def _endpoint(bucket: str) -> str:
    return f"https://{bucket}.s3.ap-south-1.amazonaws.com"


def _list(bucket: str, prefix: str, delimiter: str = "/") -> tuple[list[str], list[str]]:
    """One page of a public ListBucketV2 call: (common prefixes, object keys).

    ponytail: single page only (max 1000 entries). Enough for the metadata/
    tree, which is all we list. Add a continuation-token loop if you ever need
    to enumerate data/pdf/ directly.
    """
    resp = requests.get(
        _endpoint(bucket),
        params={"list-type": "2", "prefix": prefix, "delimiter": delimiter,
                "max-keys": "1000"},
        timeout=60,
    )
    resp.raise_for_status()
    tree = ET.fromstring(resp.text)
    prefixes = [p.findtext(S3_NS + "Prefix") for p in tree.findall(S3_NS + "CommonPrefixes")]
    keys = [c.findtext(S3_NS + "Key") for c in tree.findall(S3_NS + "Contents")]
    return prefixes, keys


def _read_parquet(bucket: str, key: str) -> pd.DataFrame:
    resp = requests.get(f"{_endpoint(bucket)}/{key}", timeout=600)
    resp.raise_for_status()
    return pd.read_parquet(io.BytesIO(resp.content))


def _parse_years(spec: str) -> list[int]:
    """'2024' -> [2024];  '2023-2025' -> [2023, 2024, 2025]."""
    years: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = (int(x) for x in part.split("-", 1))
            years.extend(range(lo, hi + 1))
        else:
            years.append(int(part))
    return sorted(set(years))


# ---------------------------------------------------------------------------
# Per-bucket planning: turn parquet metadata into download jobs
# ---------------------------------------------------------------------------

def _plan_supreme_court(years: list[int], limit: int | None) -> list[tuple]:
    """SC layout: metadata/parquet/year=Y/metadata.parquet, one flat file.

    The PDF key is derived from the ``path`` column:
    ``2024_10_108_125`` -> ``data/pdf/year=2024/english/2024_10_108_125_EN.pdf``
    """
    plan = []
    for year in years:
        try:
            df = _read_parquet(SC_BUCKET, f"metadata/parquet/year={year}/metadata.parquet")
        except requests.HTTPError:
            print(f"  no SC metadata for {year}, skipping", file=sys.stderr)
            continue
        df = df[df["path"].notna()]
        if limit:
            df = df.head(limit)
        out_dir = OUT_ROOT / "supreme_court" / str(year)
        for row in df.to_dict("records"):
            name = f"{row['path']}_EN.pdf"
            plan.append((
                SC_BUCKET,
                f"data/pdf/year={year}/english/{name}",
                out_dir / name,
                _manifest_row(row, name, "Supreme Court of India", year),
            ))
        print(f"  SC {year}: {len(df)} judgments")
    return plan


def _plan_high_courts(years: list[int], courts: list[str] | None,
                      limit: int | None) -> list[tuple]:
    """HC data is partitioned by court and bench, so we walk the prefix tree.

    PDF key = ``data/pdf/year=Y/court=C/bench=B/<basename(pdf_link)>``. Only the
    plain ``metadata.parquet`` is used -- its ``metadata-mobile.parquet`` sibling
    indexes rows whose PDFs were never mirrored, and those objects 404.
    """
    plan = []
    for year in years:
        court_prefixes, _ = _list(HC_BUCKET, f"metadata/parquet/year={year}/")
        for court_prefix in court_prefixes:
            court = court_prefix.rstrip("/").rsplit("court=", 1)[-1]
            if courts and court not in courts:
                continue
            bench_prefixes, _ = _list(HC_BUCKET, court_prefix)
            taken = 0
            for bench_prefix in bench_prefixes:
                if limit and taken >= limit:
                    break
                bench = bench_prefix.rstrip("/").rsplit("bench=", 1)[-1]
                try:
                    df = _read_parquet(HC_BUCKET, bench_prefix + "metadata.parquet")
                except requests.HTTPError:
                    continue
                df = df[df["pdf_link"].notna()]
                if limit:
                    df = df.head(limit - taken)
                out_dir = OUT_ROOT / f"hc_{court}" / str(year)
                for row in df.to_dict("records"):
                    name = basename(str(row["pdf_link"]))
                    plan.append((
                        HC_BUCKET,
                        f"data/pdf/year={year}/court={court}/bench={bench}/{name}",
                        out_dir / name,
                        _manifest_row(row, name, row.get("court") or f"HC {court}", year),
                    ))
                taken += len(df)
            if taken:
                print(f"  HC {court} {year}: {taken} judgments")
    return plan


def _manifest_row(row: dict, pdf_name: str, court: str, year: int) -> dict:
    record = {column: "" for column in MANIFEST_COLUMNS}
    for column in MANIFEST_COLUMNS:
        value = row.get(column)
        if value is not None and not pd.isna(value):
            record[column] = str(value)
    # Set last so bucket-specific columns can never shadow our own identifiers.
    record.update(pdf=pdf_name, court=str(court), year=year)
    return record


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def _download(job: tuple) -> tuple[bool, dict | None]:
    bucket, key, dest, meta = job
    if dest.exists() and dest.stat().st_size > 0:
        return True, None  # already have it; don't duplicate the manifest row
    try:
        resp = requests.get(f"{_endpoint(bucket)}/{key}", timeout=300)
        resp.raise_for_status()
    except requests.RequestException:
        # The mirror is sparse: a fraction of metadata rows point at PDFs that
        # were never archived. Skipping them is expected, not a run failure.
        return False, None
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(resp.content)
    return True, meta


def _run(plan: list[tuple], workers: int) -> None:
    if not plan:
        print("Nothing to download.")
        return
    print(f"Downloading {len(plan)} PDFs with {workers} workers...")
    by_dir: dict[Path, list[dict]] = {}
    ok = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for done, (success, meta) in enumerate(pool.map(_download, plan), start=1):
            if success:
                ok += 1
            if meta:
                by_dir.setdefault(plan[done - 1][2].parent, []).append(meta)
            if done % 100 == 0:
                print(f"  {done}/{len(plan)}  ({ok} ok)")

    for directory, rows in by_dir.items():
        manifest = directory / "manifest.csv"
        write_header = not manifest.exists()
        with manifest.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
            if write_header:
                writer.writeheader()
            writer.writerows(rows)
    print(f"Done: {ok}/{len(plan)} PDFs under {OUT_ROOT}/  "
          f"({len(plan) - ok} not mirrored)")


def _list_courts(year: int) -> None:
    prefixes, _ = _list(HC_BUCKET, f"metadata/parquet/year={year}/")
    codes = sorted(p.rstrip("/").rsplit("court=", 1)[-1] for p in prefixes)
    print(f"High Court codes present for {year}:")
    print("  " + "  ".join(codes))
    print("\nCode -> court name mapping: "
          "https://github.com/vanga/indian-high-court-judgments")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)

    for name, help_text in [("sc", "Supreme Court (reported SCR judgments)"),
                            ("hc", "High Courts")]:
        cmd = sub.add_parser(name, help=help_text)
        cmd.add_argument("--years", default="2024", help='e.g. "2024" or "2020-2024"')
        cmd.add_argument("--limit", type=int, default=100,
                         help="max judgments per court-year (0 = no cap)")
        cmd.add_argument("--workers", type=int, default=8)
        if name == "hc":
            cmd.add_argument("--courts", default=None,
                             help="comma-separated codes, e.g. 27_1,1_12 (default: all)")

    courts_cmd = sub.add_parser("courts", help="list available High Court codes")
    courts_cmd.add_argument("--year", type=int, default=2024)

    args = parser.parse_args()

    if args.command == "courts":
        _list_courts(args.year)
        return

    years = _parse_years(args.years)
    limit = args.limit or None
    print(f"Planning {args.command.upper()} fetch for years {years}...")
    if args.command == "sc":
        plan = _plan_supreme_court(years, limit)
    else:
        courts = [c.strip() for c in args.courts.split(",")] if args.courts else None
        plan = _plan_high_courts(years, courts, limit)
    _run(plan, args.workers)


if __name__ == "__main__":
    main()
