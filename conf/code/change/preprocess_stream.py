import argparse
import glob
import json
import os
import pickle
import random
import time
from collections import Counter, defaultdict

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


def detect_encoding(file_path: str) -> str:
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
                sample = f.readline()
                if sample:
                    json.loads(sample.strip())
                return enc
        except Exception:
            continue
    raise ValueError(f"Cannot detect a usable encoding for {file_path}")


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


def stream_analyze(file_path, max_lines=None, sample_ratio=None):
    print(f"Analyzing: {file_path}")
    stats = {
        "total_reviews": 0,
        "users": set(),
        "items": set(),
        "ratings": Counter(),
        "text_lengths": [],
        "user_review_count": Counter(),
    }

    start_time = time.time()
    enc = detect_encoding(file_path)
    with open(file_path, "r", encoding=enc) as f:
        for i, line in enumerate(tqdm(f, desc="Analyze")):
            if max_lines is not None and i >= max_lines:
                break
            if sample_ratio is not None and random.random() > sample_ratio:
                continue
            try:
                review = json.loads(line.strip())
            except json.JSONDecodeError:
                continue

            stats["total_reviews"] += 1
            user_id = review.get("user_id") or review.get("reviewerID")
            if user_id:
                stats["users"].add(user_id)
                stats["user_review_count"][user_id] += 1

            item_id = review.get("asin") or review.get("parent_asin")
            if item_id:
                stats["items"].add(item_id)

            rating = review.get("rating", review.get("overall"))
            if rating is not None:
                stats["ratings"][rating] += 1

            text = review.get("text") or review.get("reviewText")
            if text:
                stats["text_lengths"].append(len(text))

            if (i + 1) % 100000 == 0:
                elapsed = time.time() - start_time
                print(f"Processed {i + 1} lines, elapsed {elapsed:.1f}s")
    return stats


def stream_generate_embeddings(
    file_path,
    model_name="D:/models/all-MiniLM-L6-v2",
    min_reviews=3,
    max_users=10000,
    max_lines=None,
    max_texts_per_user=10,
    max_text_len=5000,
):
    print(f"Loading model: {model_name}")
    model = SentenceTransformer(model_name)

    user_texts = defaultdict(list)
    enc = detect_encoding(file_path)
    with open(file_path, "r", encoding=enc) as f:
        for i, line in enumerate(tqdm(f, desc="Collect user texts")):
            if max_lines is not None and i >= max_lines:
                break
            try:
                review = json.loads(line.strip())
            except Exception:
                continue

            user_id = review.get("user_id") or review.get("reviewerID")
            if not user_id:
                continue
            text = review.get("text") or review.get("reviewText")
            if not text:
                continue
            title = review.get("title") or review.get("summary") or ""
            merged = f"{title} {text}".strip()
            if len(user_texts[user_id]) < max_texts_per_user:
                user_texts[user_id].append(merged)

    valid_users = []
    valid_texts = []
    for user_id, texts in user_texts.items():
        if len(texts) >= min_reviews:
            combined = " ".join(texts[:max_texts_per_user])
            if len(combined) > max_text_len:
                combined = combined[:max_text_len]
            valid_users.append(user_id)
            valid_texts.append(combined)

    if max_users is not None and len(valid_users) > max_users:
        idx = list(range(len(valid_users)))
        random.shuffle(idx)
        idx = idx[:max_users]
        valid_users = [valid_users[i] for i in idx]
        valid_texts = [valid_texts[i] for i in idx]

    if not valid_users:
        raise RuntimeError("No valid users found for embedding generation.")

    embeddings = []
    batch_size = 32
    for i in tqdm(range(0, len(valid_texts), batch_size), desc="Encode users"):
        batch = valid_texts[i : i + batch_size]
        batch_emb = model.encode(batch, normalize_embeddings=True, show_progress_bar=False)
        embeddings.append(batch_emb)
    embeddings = np.vstack(embeddings).astype(np.float32)
    return embeddings, valid_users


def main():
    parser = argparse.ArgumentParser(description="Analyze reviews and generate user embeddings.")
    parser.add_argument("--dataset", type=str, default="amazon")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--reviews_file", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--mode", choices=["analyze", "generate", "both"], default="analyze")
    parser.add_argument("--max_lines", type=int, default=None)
    parser.add_argument("--sample_ratio", type=float, default=None)
    parser.add_argument("--max_users", type=int, default=10000)
    parser.add_argument("--min_reviews", type=int, default=3)
    parser.add_argument("--max_texts_per_user", type=int, default=10)
    parser.add_argument("--max_text_len", type=int, default=5000)
    parser.add_argument("--model", type=str, default="D:/models/all-MiniLM-L6-v2")
    args = parser.parse_args()

    dataset_dir = os.path.join(args.data_dir, args.dataset)
    reviews_file = args.reviews_file or resolve_default_reviews_file(dataset_dir)
    output_dir = args.output_dir or dataset_dir

    if args.mode in ["analyze", "both"]:
        stats = stream_analyze(
            reviews_file,
            max_lines=args.max_lines,
            sample_ratio=args.sample_ratio,
        )
        print(f"reviews={stats['total_reviews']}")
        print(f"users={len(stats['users'])}")
        print(f"items={len(stats['items'])}")
        if stats["text_lengths"]:
            print(f"avg_text_len={np.mean(stats['text_lengths']):.1f}")
            print(f"median_text_len={np.median(stats['text_lengths']):.1f}")

    if args.mode in ["generate", "both"]:
        embeddings, user_ids = stream_generate_embeddings(
            reviews_file,
            model_name=args.model,
            min_reviews=args.min_reviews,
            max_users=args.max_users,
            max_lines=args.max_lines,
            max_texts_per_user=args.max_texts_per_user,
            max_text_len=args.max_text_len,
        )
        os.makedirs(output_dir, exist_ok=True)
        usr_emb_path = os.path.join(output_dir, "usr_emb_np.pkl")
        with open(usr_emb_path, "wb") as f:
            pickle.dump(embeddings, f)
        user_ids_path = os.path.join(output_dir, "user_ids.pkl")
        with open(user_ids_path, "wb") as f:
            pickle.dump(user_ids, f)
        meta_path = os.path.join(output_dir, "meta_info.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "num_users": len(user_ids),
                    "embedding_dim": int(embeddings.shape[1]),
                    "min_reviews": args.min_reviews,
                    "source_file": reviews_file,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
        print(f"Saved: {usr_emb_path} {embeddings.shape}")
        print(f"Saved: {user_ids_path} ({len(user_ids)})")


if __name__ == "__main__":
    main()
