"""Offline checks for the judgment fetcher's key-derivation and planning logic.

Network calls are stubbed -- the only things worth guarding here are the bits
that silently produce 404s if they drift: year parsing, S3 key construction,
and manifest rows never losing their own identifier columns to a bucket column
of the same name.
"""

import io
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import fetch_judgments as fj


def test_parse_years():
    assert fj._parse_years("2024") == [2024]
    assert fj._parse_years("2023-2025") == [2023, 2024, 2025]
    assert fj._parse_years("2020,2023-2024") == [2020, 2023, 2024]
    assert fj._parse_years("2024,2024") == [2024]  # deduped


def test_supreme_court_key_derivation(monkeypatch):
    df = pd.DataFrame([
        {"path": "2024_10_108_125", "title": "VIJAY SINGH versus STATE OF BIHAR",
         "citation": "[2024] 10 S.C.R. 108", "decision_date": "25-09-2024"},
        {"path": None, "title": "unmirrored"},  # must be filtered out
    ])
    monkeypatch.setattr(fj, "_read_parquet", lambda bucket, key: df)

    plan = fj._plan_supreme_court([2024], limit=None)

    assert len(plan) == 1
    bucket, key, dest, meta = plan[0]
    assert bucket == fj.SC_BUCKET
    assert key == "data/pdf/year=2024/english/2024_10_108_125_EN.pdf"
    assert dest == fj.OUT_ROOT / "supreme_court" / "2024" / "2024_10_108_125_EN.pdf"
    assert meta["citation"] == "[2024] 10 S.C.R. 108"
    assert meta["court"] == "Supreme Court of India"


def test_high_court_key_derivation(monkeypatch):
    # pdf_link is a full URL path; only its basename is the real object name.
    df = pd.DataFrame([
        {"pdf_link": "court/cnrorders/newos/orders/HCBM020000782017_1_2024-01-17.pdf",
         "cnr": "HCBM020000782017", "court": "Bombay High Court",
         "decision_date": "2024-01-17"},
    ])
    monkeypatch.setattr(fj, "_read_parquet", lambda bucket, key: df)
    monkeypatch.setattr(fj, "_list", lambda bucket, prefix, delimiter="/": (
        ["metadata/parquet/year=2024/court=27_1/"] if prefix.endswith("year=2024/")
        else ["metadata/parquet/year=2024/court=27_1/bench=newos/"], []
    ))

    plan = fj._plan_high_courts([2024], courts=["27_1"], limit=None)

    assert len(plan) == 1
    bucket, key, dest, meta = plan[0]
    assert bucket == fj.HC_BUCKET
    assert key == ("data/pdf/year=2024/court=27_1/bench=newos/"
                   "HCBM020000782017_1_2024-01-17.pdf")
    assert dest.parent == fj.OUT_ROOT / "hc_27_1" / "2024"
    assert meta["court"] == "Bombay High Court"


def test_high_court_respects_court_filter(monkeypatch):
    monkeypatch.setattr(fj, "_list", lambda bucket, prefix, delimiter="/": (
        ["metadata/parquet/year=2024/court=27_1/",
         "metadata/parquet/year=2024/court=1_12/"], []
    ))
    monkeypatch.setattr(fj, "_read_parquet",
                        lambda b, k: pytest.fail("filtered court was fetched"))

    assert fj._plan_high_courts([2024], courts=["99_9"], limit=None) == []


def test_manifest_row_keeps_own_identifiers():
    # A bucket column literally named "year"/"court" must not overwrite ours.
    row = {"year": 1999, "court": "WRONG", "pdf": "wrong.pdf", "title": "T"}
    record = fj._manifest_row(row, "right.pdf", "Supreme Court of India", 2024)

    assert record["pdf"] == "right.pdf"
    assert record["court"] == "Supreme Court of India"
    assert record["year"] == 2024
    assert record["title"] == "T"
    assert set(record) == set(fj.MANIFEST_COLUMNS)


def test_manifest_row_blanks_missing_and_nan():
    record = fj._manifest_row({"judge": None, "citation": float("nan")},
                              "x.pdf", "HC", 2024)
    assert record["judge"] == ""
    assert record["citation"] == ""


def test_download_skips_existing_without_duplicating_manifest(tmp_path):
    dest = tmp_path / "already.pdf"
    dest.write_bytes(b"%PDF-1.7")

    success, meta = fj._download((fj.SC_BUCKET, "irrelevant/key.pdf", dest, {"pdf": "x"}))

    assert success is True
    assert meta is None  # re-runs must not append the row a second time


def test_download_tolerates_unmirrored_pdf(tmp_path, monkeypatch):
    import requests

    def boom(*_args, **_kwargs):
        raise requests.HTTPError("404")

    monkeypatch.setattr(fj.requests, "get", boom)
    success, meta = fj._download(
        (fj.SC_BUCKET, "missing.pdf", tmp_path / "missing.pdf", {"pdf": "x"})
    )

    assert (success, meta) == (False, None)
    assert not (tmp_path / "missing.pdf").exists()


def test_list_parses_s3_xml(monkeypatch):
    xml = """<?xml version="1.0" encoding="UTF-8"?>
    <ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">
      <CommonPrefixes><Prefix>metadata/parquet/year=2024/</Prefix></CommonPrefixes>
      <Contents><Key>metadata/parquet/year=2024/metadata.parquet</Key></Contents>
    </ListBucketResult>"""

    class FakeResponse:
        text = xml
        def raise_for_status(self): pass

    monkeypatch.setattr(fj.requests, "get", lambda *a, **k: FakeResponse())
    prefixes, keys = fj._list(fj.SC_BUCKET, "metadata/parquet/")

    assert prefixes == ["metadata/parquet/year=2024/"]
    assert keys == ["metadata/parquet/year=2024/metadata.parquet"]


def test_read_parquet_roundtrip(monkeypatch):
    buffer = io.BytesIO()
    pd.DataFrame([{"path": "2024_1_1_2"}]).to_parquet(buffer)

    class FakeResponse:
        content = buffer.getvalue()
        def raise_for_status(self): pass

    monkeypatch.setattr(fj.requests, "get", lambda *a, **k: FakeResponse())
    assert fj._read_parquet(fj.SC_BUCKET, "k")["path"].tolist() == ["2024_1_1_2"]
