#!/usr/bin/env python3
"""Upload {stem}.js files to Cloudflare R2 (S3-compatible) and verify public GETs.

Env (required):
  R2_ACCOUNT_ID or R2_ENDPOINT
  R2_ACCESS_KEY_ID
  R2_SECRET_ACCESS_KEY
  R2_BUCKET
  CDN_PUBLIC_BASE   # public HTTPS base for the bucket, e.g. https://cdn.example.com

Usage:
  .venv/bin/python scripts/upload_r2.py runs/round24-smoke --prefix round-24-smoke
  .venv/bin/python scripts/upload_r2.py runs/round24-smoke --prefix round-24-smoke --dry-run
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import boto3
import httpx
from botocore.client import Config


def _endpoint() -> str:
    if os.environ.get("R2_ENDPOINT"):
        return os.environ["R2_ENDPOINT"].rstrip("/")
    account = os.environ.get("R2_ACCOUNT_ID")
    if not account:
        raise SystemExit("Set R2_ENDPOINT or R2_ACCOUNT_ID")
    return f"https://{account}.r2.cloudflarestorage.com"


def _require(name: str) -> str:
    v = os.environ.get(name)
    if not v:
        raise SystemExit(f"Missing env var: {name}")
    return v


def main() -> int:
    p = argparse.ArgumentParser(description="Upload .js files to R2 for miner CDN smoke tests")
    p.add_argument("js_dir", type=Path, help="Directory containing {stem}.js files")
    p.add_argument("--prefix", required=True, help="Object key prefix, e.g. round-24-smoke")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--skip-verify", action="store_true", help="Skip public HTTPS GET checks")
    args = p.parse_args()

    js_dir: Path = args.js_dir
    if not js_dir.is_dir():
        print(f"Not a directory: {js_dir}", file=sys.stderr)
        return 1

    files = sorted(js_dir.glob("*.js"))
    if not files:
        print(f"No .js files in {js_dir}", file=sys.stderr)
        return 1

    prefix = args.prefix.strip("/")
    cdn_base = os.environ.get("CDN_PUBLIC_BASE", "").rstrip("/")
    bucket = _require("R2_BUCKET") if not args.dry_run else os.environ.get("R2_BUCKET", "<bucket>")
    endpoint = _endpoint() if not args.dry_run else os.environ.get("R2_ENDPOINT", "<endpoint>")

    print(f"files={len(files)}  prefix={prefix}  bucket={bucket}")
    print(f"cdn_url would be: {cdn_base}/{prefix}" if cdn_base else "(set CDN_PUBLIC_BASE for public verify)")

    if args.dry_run:
        for f in files:
            key = f"{prefix}/{f.name}"
            print(f"  DRY  s3://{bucket}/{key}  ({f.stat().st_size} bytes)")
        return 0

    client = boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=_require("R2_ACCESS_KEY_ID"),
        aws_secret_access_key=_require("R2_SECRET_ACCESS_KEY"),
        config=Config(signature_version="s3v4"),
        region_name=os.environ.get("R2_REGION", "auto"),
    )

    uploaded = 0
    for f in files:
        key = f"{prefix}/{f.name}"
        body = f.read_bytes()
        client.put_object(
            Bucket=bucket,
            Key=key,
            Body=body,
            ContentType="application/javascript",
        )
        print(f"  OK  {key} ({len(body)} bytes)")
        uploaded += 1

    print(f"uploaded {uploaded}/{len(files)}")

    if args.skip_verify or not cdn_base:
        if not cdn_base:
            print("skip verify: CDN_PUBLIC_BASE unset")
        return 0

    cdn_url = f"{cdn_base}/{prefix}"
    fails = 0
    with httpx.Client(timeout=30.0, follow_redirects=True) as http:
        for f in files[: min(5, len(files))]:
            url = f"{cdn_url}/{f.name}"
            r = http.head(url)
            if r.status_code >= 400:
                # some CDNs block HEAD
                r = http.get(url)
            ok = r.status_code == 200
            print(f"  {'OK' if ok else 'FAIL'}  GET {url} -> {r.status_code}")
            if not ok:
                fails += 1

    print(f"cdn_url={cdn_url}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
