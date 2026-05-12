import argparse
import json
import os
import pickle
import random
import time
from collections import Counter, defaultdict

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


def analyze_reviews(file_path, max_lines=None, sample_ratio=None):
    print(f"Analyze reviews: {file_path}")
    stats = {
        "total_reviews": 0,
        "users": set(),
        "items": set(),
        "ratings": Counter(),
        "text_lengths": [],
        "user_review_count": Counter(),
    }

    start_time = time.time()
    with open(file_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(tqdm(f, desc="Analyze")):
            if max_lines and i >= max_lines:
                break
            if sample_ratio and random.random() > sample_ratio:
                continue
            try:
                r = json.loads(line.strip())
            except json.JSONDecodeError:
                continue

            stats["total_reviews"] += 1
            uid = r.get("user_id") or r.get("reviewerID")
            if uid:
                stats["users"].add(uid)
                stats["user_review_count"][uid] += 1
            asin = r.get("asin") or r.get("parent_asin")
            if asin:
                stats["items"].add(asin)
            rating = r.get("rating") or r.get("overall")
            if rating is not None:
                stats["ratings"][rating] += 1
            text = r.get("text") or r.get("reviewText")
            if text:
                stats["text_lengths"].append(len(text))

            if (i + 1) % 100000 == 0:
                elapsed = time.time() - start_time
                print(f"Processed {i + 1} lines in {elapsed:.1f}s")

    return stats


def generate_user_embeddings_from_reviews(
    file_path: str,
    output_dir: str = "data/digital_music",
    model_name: str = "D:/models/all-MiniLM-L6-v2",
    min_reviews: int = 3,
    max_users: int | None = 10000,
    max_lines: int | None = None,
    max_texts_per_user: int = 10,
    max_text_len: int = 5000,
):
    print(f"Load model: {model_name}")
    model = SentenceTransformer(model_name)
    user_texts = defaultdict(list)

    with open(file_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(tqdm(f, desc="Collect user texts")):
            if max_lines and i >= max_lines:
                break
            try:
                r = json.loads(line.strip())
            except Exception:
                continue

            uid = r.get("user_id") or r.get("reviewerID")
            if not uid:
                continue
            text = r.get("text") or r.get("reviewText") or ""
            title = r.get("title") or r.get("summary") or ""
            merged = f"{title} {text}".strip()
            if not merged:
                continue
            if len(user_texts[uid]) < max_texts_per_user:
                user_texts[uid].append(merged)

    valid_users = []
    valid_texts = []
    for uid, texts in user_texts.items():
        if len(texts) >= min_reviews:
            combined = " ".join(texts[:max_texts_per_user])
            if len(combined) > max_text_len:
                combined = combined[:max_text_len]
            valid_users.append(uid)
            valid_texts.append(combined)

    if max_users is not None and max_users > 0 and len(valid_users) > max_users:
        idx = list(range(len(valid_users)))
        random.shuffle(idx)
        idx = idx[:max_users]
        valid_users = [valid_users[i] for i in idx]
        valid_texts = [valid_texts[i] for i in idx]

    if not valid_users:
        raise RuntimeError("No valid users found for embedding generation.")

    batch_size = 32
    embeddings = []
    for i in tqdm(range(0, len(valid_texts), batch_size), desc="Encode users"):
        batch = valid_texts[i : i + batch_size]
        batch_emb = model.encode(batch, normalize_embeddings=True, show_progress_bar=False)
        embeddings.append(batch_emb)
    embeddings = np.vstack(embeddings).astype(np.float32)

    os.makedirs(output_dir, exist_ok=True)
    usr_emb_path = os.path.join(output_dir, "usr_emb_np.pkl")
    user_ids_path = os.path.join(output_dir, "user_ids.pkl")

    with open(usr_emb_path, "wb") as f:
        pickle.dump(embeddings, f)
    with open(user_ids_path, "wb") as f:
        pickle.dump(valid_users, f)

    print(f"Saved: {usr_emb_path}, shape={embeddings.shape}")
    print(f"Saved: {user_ids_path}, count={len(valid_users)}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate user semantic embeddings from reviews."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="digital_music",
        help="Dataset name for default path resolution.",
    )
    parser.add_argument(
        "--reviews_path",
        type=str,
        default=None,
        help="Reviews path. Default: data/<dataset>/reviews_clean.jsonl",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory. Default: data/<dataset>",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="D:/models/all-MiniLM-L6-v2",
        help="SentenceTransformer model path/name.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["analyze", "generate", "both"],
        default="both",
        help="Run analyze / generate / both.",
    )
    parser.add_argument("--max_lines", type=int, default=None)
    parser.add_argument("--sample_ratio", type=float, default=None)
    parser.add_argument("--max_users", type=int, default=10000)
    parser.add_argument("--min_reviews", type=int, default=3)
    parser.add_argument("--max_texts_per_user", type=int, default=10)
    parser.add_argument("--max_text_len", type=int, default=5000)
    args = parser.parse_args()

    reviews_path = args.reviews_path or os.path.join(
        "data", args.dataset, "reviews_clean.jsonl"
    )
    output_dir = args.output_dir or os.path.join("data", args.dataset)

    if args.mode in ["analyze", "both"]:
        stats = analyze_reviews(
            reviews_path,
            max_lines=args.max_lines,
            sample_ratio=args.sample_ratio,
        )
        print(f"reviews={stats['total_reviews']}, users={len(stats['users'])}, items={len(stats['items'])}")

    if args.mode in ["generate", "both"]:
        max_users = args.max_users if args.max_users and args.max_users > 0 else None
        generate_user_embeddings_from_reviews(
            file_path=reviews_path,
            output_dir=output_dir,
            model_name=args.model,
            min_reviews=args.min_reviews,
            max_users=max_users,
            max_lines=args.max_lines,
            max_texts_per_user=args.max_texts_per_user,
            max_text_len=args.max_text_len,
        )


if __name__ == "__main__":
    main()
