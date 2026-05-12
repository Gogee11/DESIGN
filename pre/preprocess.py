import argparse
import json
import os
import pickle
from collections import defaultdict

import numpy as np
from scipy.sparse import coo_matrix


def parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocess Amazon review JSON into project format."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="digital_music",
        help="Dataset name. Output path: data/<dataset>/",
    )
    parser.add_argument(
        "--src",
        type=str,
        default=None,
        help="Source json path. Default: data/<dataset>/<dataset>.json",
    )
    parser.add_argument(
        "--min_rating",
        type=float,
        default=1.0,
        help="Keep interactions with rating >= min_rating.",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.6,
        help="Train split ratio.",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.2,
        help="Validation split ratio.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2025,
        help="Random seed for split.",
    )
    return parser.parse_args()


def unique_ordered(seq):
    seen = set()
    res = []
    for x in seq:
        if x not in seen:
            seen.add(x)
            res.append(x)
    return res


def main():
    args = parse_args()

    src_path = args.src or os.path.join("data", args.dataset, f"{args.dataset}.json")
    out_dir = os.path.join("data", args.dataset)
    os.makedirs(out_dir, exist_ok=True)

    users = set()
    items = set()
    rows_u, rows_i = [], []

    reviews_out_path = os.path.join(out_dir, "reviews_clean.jsonl")
    total_lines = 0
    valid_interactions = 0

    with open(reviews_out_path, "w", encoding="utf-8") as fout:
        with open(src_path, "r", encoding="utf-8") as f:
            for line in f:
                total_lines += 1
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue

                user_id = r.get("user_id") or r.get("reviewerID")
                asin = r.get("asin") or r.get("parent_asin")
                rating = float(r.get("rating", r.get("overall", 0.0)))
                text = r.get("text", r.get("reviewText", "")) or ""
                title = r.get("title", r.get("summary", "")) or ""

                if not user_id or not asin:
                    continue

                clean_rec = {
                    "user_id": user_id,
                    "asin": asin,
                    "parent_asin": asin,
                    "rating": rating,
                    "text": text,
                    "title": title,
                }
                fout.write(json.dumps(clean_rec, ensure_ascii=False) + "\n")

                if rating >= args.min_rating:
                    users.add(user_id)
                    items.add(asin)
                    rows_u.append(user_id)
                    rows_i.append(asin)
                    valid_interactions += 1

    if not users or not items:
        raise RuntimeError("No valid users or items found.")

    user_ids = unique_ordered(rows_u)
    item_ids = unique_ordered(rows_i)
    user2id = {u: i for i, u in enumerate(user_ids)}
    item2id = {it: i for i, it in enumerate(item_ids)}
    num_users = len(user_ids)
    num_items = len(item_ids)

    with open(os.path.join(out_dir, "user_ids.pkl"), "wb") as f:
        pickle.dump(user_ids, f)
    with open(os.path.join(out_dir, "item_ids.pkl"), "wb") as f:
        pickle.dump(item_ids, f)

    int_rows, int_cols = [], []
    for u, i in zip(rows_u, rows_i):
        if u in user2id and i in item2id:
            int_rows.append(user2id[u])
            int_cols.append(item2id[i])

    user_items = defaultdict(set)
    for r, c in zip(int_rows, int_cols):
        user_items[r].add(c)

    np.random.seed(args.seed)
    trn_r, trn_c = [], []
    val_r, val_c = [], []
    tst_r, tst_c = [], []
    t2 = args.train_ratio
    t3 = args.train_ratio + args.val_ratio

    for u in range(num_users):
        items_u = list(user_items.get(u, set()))
        if not items_u:
            continue
        perm = np.random.permutation(items_u)
        n = len(perm)
        e1 = int(n * t2)
        e2 = int(n * t3)
        for j, it in enumerate(perm):
            if j < e1:
                trn_r.append(u)
                trn_c.append(it)
            elif j < e2:
                val_r.append(u)
                val_c.append(it)
            else:
                tst_r.append(u)
                tst_c.append(it)

    trn_mat = coo_matrix(
        (np.ones(len(trn_r)), (trn_r, trn_c)), shape=(num_users, num_items)
    )
    val_mat = coo_matrix(
        (np.ones(len(val_r)), (val_r, val_c)), shape=(num_users, num_items)
    )
    tst_mat = coo_matrix(
        (np.ones(len(tst_r)), (tst_r, tst_c)), shape=(num_users, num_items)
    )

    trn_edges = set(zip(trn_r, trn_c))
    val_edges = set(zip(val_r, val_c))
    tst_edges = set(zip(tst_r, tst_c))
    if trn_edges & val_edges or trn_edges & tst_edges or val_edges & tst_edges:
        raise RuntimeError("Split overlap detected after preprocessing.")

    with open(os.path.join(out_dir, "trn_mat.pkl"), "wb") as f:
        pickle.dump(trn_mat, f)
    with open(os.path.join(out_dir, "val_mat.pkl"), "wb") as f:
        pickle.dump(val_mat, f)
    with open(os.path.join(out_dir, "tst_mat.pkl"), "wb") as f:
        pickle.dump(tst_mat, f)

    stats = {
        "num_users": num_users,
        "num_items": num_items,
        "total_lines": total_lines,
        "valid_interactions": valid_interactions,
        "trn_interactions": int(trn_mat.nnz),
        "val_interactions": int(val_mat.nnz),
        "tst_interactions": int(tst_mat.nnz),
        "src_file": src_path,
        "min_rating": args.min_rating,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "seed": args.seed,
    }
    with open(os.path.join(out_dir, "data_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"Done. Dataset={args.dataset}, users={num_users}, items={num_items}")


if __name__ == "__main__":
    main()
