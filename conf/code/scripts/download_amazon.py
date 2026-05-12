"""Download an Amazon review category from McAuley-Lab/Amazon-Reviews-2023 (HF) or
fall back to the public McAuley 2018 mirror, and save as a one-per-line JSONL
file at data/<dataset>/<dataset>.jsonl ready for scripts/prepare_dataset.py.

Usage:
    python -m scripts.download_amazon --category Luxury_Beauty --dataset luxury_beauty
    python -m scripts.download_amazon --category Industrial_and_Scientific --dataset industrial_scientific
"""
import argparse
import gzip
import io
import json
import os
import sys
import urllib.request


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--category', type=str, required=True,
                   help="Amazon category, e.g., Luxury_Beauty, Industrial_and_Scientific")
    p.add_argument('--dataset', type=str, required=True,
                   help="Output dataset directory name under data/")
    p.add_argument('--source', type=str,
                   choices=['hf2023', 'mcauley2018', 'auto'],
                   default='auto')
    p.add_argument('--out_dir', type=str, default=None)
    p.add_argument('--max_lines', type=int, default=None,
                   help="Cap on number of lines for quick smoke tests")
    return p.parse_args()


def download_via_hf(category, out_path, max_lines=None):
    print(f"[hf] loading McAuley-Lab/Amazon-Reviews-2023 :: raw_review_{category}")
    from datasets import load_dataset
    ds = load_dataset(
        "McAuley-Lab/Amazon-Reviews-2023",
        f"raw_review_{category}",
        split="full",
        trust_remote_code=True,
    )
    n = 0
    with open(out_path, 'w', encoding='utf-8') as f:
        for row in ds:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
            n += 1
            if max_lines and n >= max_lines:
                break
            if n % 100_000 == 0:
                print(f"[hf] wrote {n}")
    print(f"[hf] saved {n} reviews -> {out_path}")


def download_via_mcauley_2018(category, out_path, max_lines=None):
    """Old McAuley 2018 mirror; URLs may rot. Used as a fallback."""
    candidates = [
        f"https://jmcauley.ucsd.edu/data/amazon_v2/categoryFilesSmall/{category}_5.json.gz",
        f"https://snap.stanford.edu/data/amazon/productGraphV2/categoryFilesSmall/{category}_5.json.gz",
        f"https://snap.stanford.edu/data/amazon/categoryFilesSmall/{category}_5.json.gz",
    ]
    last_err = None
    for url in candidates:
        try:
            print(f"[mcauley] trying {url}")
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = resp.read()
            print(f"[mcauley] got {len(data)} bytes")
            n = 0
            with gzip.GzipFile(fileobj=io.BytesIO(data)) as gz:
                with open(out_path, 'w', encoding='utf-8') as f:
                    for line in gz:
                        f.write(line.decode('utf-8'))
                        n += 1
                        if max_lines and n >= max_lines:
                            break
            print(f"[mcauley] saved {n} reviews -> {out_path}")
            return
        except Exception as e:
            last_err = e
            print(f"[mcauley] {url} failed: {e}")
    raise RuntimeError(f"All McAuley 2018 URLs failed; last={last_err}")


def main():
    args = parse_args()
    out_dir = args.out_dir or os.path.join('data', args.dataset)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{args.dataset}.jsonl")

    if args.source == 'hf2023':
        download_via_hf(args.category, out_path, args.max_lines)
    elif args.source == 'mcauley2018':
        download_via_mcauley_2018(args.category, out_path, args.max_lines)
    else:
        try:
            download_via_hf(args.category, out_path, args.max_lines)
        except Exception as e:
            print(f"[auto] hf failed ({e}); falling back to mcauley 2018")
            download_via_mcauley_2018(args.category, out_path, args.max_lines)


if __name__ == '__main__':
    main()
