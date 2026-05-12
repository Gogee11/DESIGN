import argparse
import glob
import json
import os
import pickle
from collections import defaultdict

import numpy as np
from scipy.sparse import coo_matrix


def try_encodings(file_path: str) -> str:
    encodings_to_try = [
        "utf-8-sig",
        "utf-8",
        "utf-16",
        "utf-16-le",
        "utf-16-be",
        "gbk",
        "latin-1",
    ]
    for enc in encodings_to_try:
        try:
            with open(file_path, "r", encoding=enc) as f:
                line = f.readline()
                if line:
                    json.loads(line.strip())
                    return enc
        except (UnicodeError, json.JSONDecodeError):
            continue
    raise ValueError(f"No working encoding found for: {file_path}")


def resolve_default_reviews_file(dataset_dir: str) -> str:
    candidates = [
        os.path.join(dataset_dir, "reviews_clean.jsonl"),
        os.path.join(dataset_dir, "reviews.jsonl"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path

    matched = sorted(glob.glob(os.path.join(dataset_dir, "*.jsonl")))
    matched = [p for p in matched if "meta" not in os.path.basename(p).lower()]
    if matched:
        return matched[0]

    raise FileNotFoundError(
        f"Cannot find reviews file under {dataset_dir}. "
        "Please pass --reviews_file explicitly."
    )


def generate_matrices(
    user_ids_path: str,
    item_ids_path: str,
    reviews_file: str,
    output_dir: str,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    seed: int = 2024,
):
    with open(user_ids_path, "rb") as f:
        user_ids = pickle.load(f)
    user2id = {uid: i for i, uid in enumerate(user_ids)}

    with open(item_ids_path, "rb") as f:
        item_ids = pickle.load(f)
    item2id = {iid: i for i, iid in enumerate(item_ids)}

    num_users = len(user_ids)
    num_items = len(item_ids)

    enc = try_encodings(reviews_file)
    rows, cols = [], []
    parse_errors = 0
    success_count = 0
    total_lines = 0

    with open(reviews_file, "r", encoding=enc) as f:
        for i, line in enumerate(f):
            total_lines += 1
            try:
                line = line.strip()
                if not line:
                    continue
                review = json.loads(line)
            except json.JSONDecodeError:
                parse_errors += 1
                continue

            user_id = review.get("user_id") or review.get("reviewerID")
            item_id = review.get("asin") or review.get("parent_asin")
            if user_id in user2id and item_id in item2id:
                rows.append(user2id[user_id])
                cols.append(item2id[item_id])
                success_count += 1

            if (i + 1) % 100000 == 0:
                print(f"Processed {i + 1} lines...")

    if not rows:
        raise RuntimeError("No interactions extracted from reviews.")

    user_items = defaultdict(list)
    for r, c in zip(rows, cols):
        user_items[r].append(c)

    np.random.seed(seed)
    trn_rows, trn_cols = [], []
    val_rows, val_cols = [], []
    tst_rows, tst_cols = [], []

    train_end_ratio = train_ratio
    val_end_ratio = train_ratio + val_ratio

    for u in range(num_users):
        items = user_items.get(u, [])
        if not items:
            continue
        shuffled = np.random.permutation(items)
        n = len(shuffled)
        trn_end = int(n * train_end_ratio)
        val_end = int(n * val_end_ratio)
        for idx, item in enumerate(shuffled):
            if idx < trn_end:
                trn_rows.append(u)
                trn_cols.append(item)
            elif idx < val_end:
                val_rows.append(u)
                val_cols.append(item)
            else:
                tst_rows.append(u)
                tst_cols.append(item)

    trn_mat = coo_matrix(
        (np.ones(len(trn_rows)), (trn_rows, trn_cols)),
        shape=(num_users, num_items),
    )
    val_mat = coo_matrix(
        (np.ones(len(val_rows)), (val_rows, val_cols)),
        shape=(num_users, num_items),
    )
    tst_mat = coo_matrix(
        (np.ones(len(tst_rows)), (tst_rows, tst_cols)),
        shape=(num_users, num_items),
    )

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "trn_mat.pkl"), "wb") as f:
        pickle.dump(trn_mat, f)
    with open(os.path.join(output_dir, "val_mat.pkl"), "wb") as f:
        pickle.dump(val_mat, f)
    with open(os.path.join(output_dir, "tst_mat.pkl"), "wb") as f:
        pickle.dump(tst_mat, f)

    stats = {
        "num_users": num_users,
        "num_items": num_items,
        "trn_interactions": int(trn_mat.nnz),
        "val_interactions": int(val_mat.nnz),
        "tst_interactions": int(tst_mat.nnz),
        "encoding_used": enc,
        "total_lines": total_lines,
        "successful_parses": success_count,
        "parse_errors": parse_errors,
        "reviews_file": reviews_file,
        "user_ids_file": user_ids_path,
        "item_ids_file": item_ids_path,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "seed": seed,
    }
    with open(os.path.join(output_dir, "data_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"Train: {trn_mat.shape}, nnz={trn_mat.nnz}")
    print(f"Val:   {val_mat.shape}, nnz={val_mat.nnz}")
    print(f"Test:  {tst_mat.shape}, nnz={tst_mat.nnz}")


def main():
    parser = argparse.ArgumentParser(description="Generate train/val/test interaction matrices.")
    parser.add_argument("--dataset", type=str, default="amazon")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--user_ids", type=str, default=None)
    parser.add_argument("--item_ids", type=str, default=None)
    parser.add_argument("--reviews_file", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--train_ratio", type=float, default=0.6)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=2024)
    args = parser.parse_args()

    dataset_dir = os.path.join(args.data_dir, args.dataset)
    user_ids_path = args.user_ids or os.path.join(dataset_dir, "user_ids.pkl")
    item_ids_path = args.item_ids or os.path.join(dataset_dir, "item_ids.pkl")
    reviews_file = args.reviews_file or resolve_default_reviews_file(dataset_dir)
    output_dir = args.output_dir or dataset_dir

    generate_matrices(
        user_ids_path=user_ids_path,
        item_ids_path=item_ids_path,
        reviews_file=reviews_file,
        output_dir=output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
