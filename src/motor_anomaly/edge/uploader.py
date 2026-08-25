"""
Ship the spool to S3 so SageMaker has something to retrain on.

    python -m motor_anomaly.edge.uploader --dry-run     # default, no AWS needed
    python -m motor_anomaly.edge.uploader --commit

Defaults to dry-run deliberately. I have been billed by AWS for a script I ran
"just to see what it does" exactly once, and once was enough.

Key layout:
    s3://<bucket>/raw/feature_version=3/dt=2026-08-25/<device>-<epoch>.jsonl

Hive-style partitioning because that's what Athena and SageMaker Processing
both expect, and it means a retrain job can select "last 90 days, feature
version 3" with a prefix filter instead of scanning the whole bucket.

The feature_version partition is not decoration. If features.py changes, old
records describe a different 46-dimensional space and mixing them silently
poisons the retrain. Partitioning on it makes that mistake structurally hard.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import socket
import time
from datetime import datetime, timezone
from pathlib import Path

from ..config import load_config
from .runner import FEATURE_VERSION


def read_spool(path: Path) -> list[dict]:
    """Tolerant reader. A half-written final line is expected after power loss
    -- skip it rather than losing the whole file."""
    if not path.exists():
        return []
    records, bad = [], 0
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                bad += 1
    if bad:
        print(f"  skipped {bad} malformed line(s) -- likely a truncated write")
    return records


def partition_key(prefix: str, device_id: str, records: list[dict]) -> str:
    ts = records[0]["ts"] if records else time.time()
    day = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")
    ver = records[0].get("feature_version", FEATURE_VERSION) if records else FEATURE_VERSION
    return f"{prefix}/feature_version={ver}/dt={day}/{device_id}-{int(ts)}.jsonl.gz"


def upload(
    records: list[dict], bucket: str, key: str, region: str, commit: bool
) -> tuple[int, int]:
    payload = gzip.compress(
        "\n".join(json.dumps(r, separators=(",", ":")) for r in records).encode()
    )
    if not commit:
        print(f"  DRY RUN: would put s3://{bucket}/{key}  ({len(payload):,} bytes gzipped)")
        return len(records), len(payload)

    import boto3  # imported lazily so dry-run works without boto3 installed

    boto3.client("s3", region_name=region).put_object(
        Bucket=bucket,
        Key=key,
        Body=payload,
        ContentEncoding="gzip",
        ContentType="application/x-ndjson",
    )
    print(f"  uploaded s3://{bucket}/{key}  ({len(payload):,} bytes)")
    return len(records), len(payload)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Upload spooled anomaly windows to S3")
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--commit", action="store_true", help="actually upload (default is dry-run)")
    ap.add_argument("--device-id", default=os.environ.get("DEVICE_ID", socket.gethostname()))
    ap.add_argument("--min-records", type=int, default=50, help="don't bother below this")
    ap.add_argument("--keep", action="store_true", help="don't truncate the spool after upload")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    spool_path = Path(cfg["edge"]["spool_path"])
    records = read_spool(spool_path)

    print(f"spool {spool_path}: {len(records)} records")
    if len(records) < args.min_records:
        print(f"below --min-records={args.min_records}, nothing to do")
        return 0

    if cfg["cloud"]["bucket"].startswith("CHANGEME") and args.commit:
        print("refusing to --commit with the placeholder bucket name still in config")
        return 1

    key = partition_key(cfg["cloud"]["prefix"], args.device_id, records)
    n, size = upload(records, cfg["cloud"]["bucket"], key, cfg["cloud"]["region"], args.commit)

    if args.commit and not args.keep:
        # Truncate only after a confirmed put. Losing telemetry is annoying;
        # double-counting it in a retrain is worse.
        spool_path.write_text("")
        print(f"  spool truncated ({n} records shipped)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
