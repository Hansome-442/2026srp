"""Kaggle 数据集下载工具（匿名，支持断点续传与重试）。

背景：Kaggle 的 `/api/v1/datasets/download/<owner>/<name>` 会 302 到 Google 存储的
签名直链。**必须先把签名直链解析出来再下载**，否则对 Kaggle 端点带 Range 头会返回
404，并把已下载文件写坏（本项目实测踩过这个坑）。

用法
----
python scripts/download_kaggle.py --dataset kaiyueyyds/dataset-of-student-classroom-behavior
python scripts/download_kaggle.py --dataset hejuncheung/scb3dataset-usingphone --out data/raw/downloads/scb3.zip
python scripts/download_kaggle.py --search "classroom behavior"          # 只搜索不下载
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import DATA_DIR, ensure_dir, get_logger  # noqa: E402

LOG = get_logger("kaggle")
API = "https://www.kaggle.com/api/v1"
UA = {"User-Agent": "Mozilla/5.0"}
DEFAULT_DIR = DATA_DIR / "raw" / "downloads"


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: D401
        return None


def resolve_signed(url: str) -> str:
    """把 Kaggle 下载地址解析成 Google 存储签名直链。"""
    opener = urllib.request.build_opener(_NoRedirect)
    try:
        resp = opener.open(urllib.request.Request(url, headers=UA), timeout=40)
        return resp.geturl()
    except urllib.error.HTTPError as exc:
        loc = exc.headers.get("Location")
        if loc:
            return loc
        raise


def http_json(url: str):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=40) as resp:
        return json.loads(resp.read().decode("utf-8"))


def search(keyword: str, page_size: int = 20) -> None:
    url = f"{API}/datasets/list?search={urllib.parse.quote(keyword)}&pageSize={page_size}"
    for item in http_json(url):
        size = (item.get("totalBytes") or 0) / 1e9
        print(f"{size:7.2f}GB | {item.get('ref'):<55} | {item.get('title')} | {item.get('licenseName')}")


def download(dataset: str, out_path: Path, retries: int = 30) -> bool:
    url = f"{API}/datasets/download/{dataset}"
    total = None
    for attempt in range(1, retries + 1):
        start = out_path.stat().st_size if out_path.exists() else 0
        try:
            signed = resolve_signed(url)
            headers = dict(UA)
            if start:
                headers["Range"] = f"bytes={start}-"
            with urllib.request.urlopen(urllib.request.Request(signed, headers=headers), timeout=60) as resp:
                status = resp.status
                length = int(resp.headers.get("Content-Length") or 0)
                total = start + length if status == 206 else length
                mode = "ab" if status == 206 else "wb"
                if status != 206:
                    start = 0
                LOG.info("第 %d 次尝试：status=%s 从 %.1fMB 继续，总计 %.1fMB",
                         attempt, status, start / 1e6, (total or 0) / 1e6)
                last = time.time()
                with open(out_path, mode) as fh:
                    while True:
                        chunk = resp.read(512 * 1024)
                        if not chunk:
                            break
                        fh.write(chunk)
                        if time.time() - last > 15:
                            done = out_path.stat().st_size
                            pct = (done / total * 100) if total else 0
                            LOG.info("  %.1f/%.1f MB (%.1f%%)", done / 1e6, (total or 0) / 1e6, pct)
                            last = time.time()
            done = out_path.stat().st_size
            if total and done >= total:
                LOG.info("下载完成：%s（%.1f MB）", out_path, done / 1e6)
                return True
            LOG.warning("尚未完整（%.1f/%.1f MB），继续续传", done / 1e6, (total or 0) / 1e6)
        except Exception as exc:  # noqa: BLE001
            got = out_path.stat().st_size if out_path.exists() else 0
            LOG.warning("第 %d 次异常（已有 %.1fMB）：%s", attempt, got / 1e6, exc)
            time.sleep(2)
    return False


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Kaggle 数据集搜索/下载")
    parser.add_argument("--search", type=str, default=None, help="搜索关键词")
    parser.add_argument("--dataset", type=str, default=None, help="owner/name")
    parser.add_argument("--out", type=str, default=None, help="输出文件或目录")
    parser.add_argument("--size", type=int, default=20)
    args = parser.parse_args(argv)

    if args.search:
        search(args.search, args.size)
        return 0
    if not args.dataset:
        parser.error("需要 --dataset 或 --search")

    name = args.dataset.split("/")[-1]
    if args.out:
        out = Path(args.out)
        if out.is_dir() or str(args.out).endswith(("/", "\\")) or out.suffix == "":
            ensure_dir(out)
            out = out / f"{name}.zip"
    else:
        ensure_dir(DEFAULT_DIR)
        out = DEFAULT_DIR / f"{name}.zip"
    ensure_dir(out.parent)
    LOG.info("下载 %s -> %s", args.dataset, out)
    ok = download(args.dataset, out)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
