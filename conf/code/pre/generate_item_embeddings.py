import argparse
import json
import os
import pickle
from collections import defaultdict

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


def generate_item_embeddings_from_reviews(
    reviews_path: str,
    output_dir: str = "data/digital_music",
    model_name: str = "D:/models/all-MiniLM-L6-v2",
    max_reviews_per_item: int = 100,
    max_text_len: int = 5000,
    max_items: int | None = None,
    min_reviews_per_item: int = 1,
):
    if not os.path.exists(reviews_path):
        raise FileNotFoundError(f"reviews file not found: {reviews_path}")

    item_texts = defaultdict(list)
    with open(reviews_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Collect item texts"):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue

            asin = r.get("asin") or r.get("parent_asin")
            if not asin:
                continue
            title = r.get("title") or r.get("summary") or ""
            text = r.get("text") or r.get("reviewText") or ""
            merged = f"{title} {text}".strip()
            if not merged:
                continue
            if len(item_texts[asin]) < max_reviews_per_item:
                item_texts[asin].append(merged)

    items = []
    texts = []
    for asin, reviews in item_texts.items():
        if len(reviews) < min_reviews_per_item:
            continue
        combined = " ".join(reviews[:max_reviews_per_item])
        if len(combined) > max_text_len:
            combined = combined[:max_text_len]
        items.append(asin)
        texts.append(combined)

    if max_items is not None and len(items) > max_items:
        idx = np.arange(len(items))
        np.random.shuffle(idx)
        idx = idx[:max_items]
        items = [items[i] for i in idx]
        texts = [texts[i] for i in idx]

    if not items:
        raise RuntimeError("No valid items found for embedding generation.")

    model = SentenceTransformer(model_name)
    batch_size = 64
    embs = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Encode items"):
        batch = texts[i : i + batch_size]
        batch_emb = model.encode(batch, normalize_embeddings=True)
        embs.append(batch_emb)
    embs = np.vstack(embs).astype(np.float32)

    os.makedirs(output_dir, exist_ok=True)
    emb_path = os.path.join(output_dir, "itm_emb_np.pkl")
    ids_path = os.path.join(output_dir, "item_ids.pkl")
    with open(emb_path, "wb") as f:
        pickle.dump(embs, f)
    with open(ids_path, "wb") as f:
        pickle.dump(items, f)

    print(f"Saved: {emb_path}, shape={embs.shape}")
    print(f"Saved: {ids_path}, count={len(items)}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate item semantic embeddings from reviews."
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
    parser.add_argument("--max_reviews_per_item", type=int, default=20)
    parser.add_argument("--max_text_len", type=int, default=5000)
    parser.add_argument("--max_items", type=int, default=None)
    parser.add_argument("--min_reviews_per_item", type=int, default=1)
    args = parser.parse_args()

    reviews_path = args.reviews_path or os.path.join(
        "data", args.dataset, "reviews_clean.jsonl"
    )
    output_dir = args.output_dir or os.path.join("data", args.dataset)

    generate_item_embeddings_from_reviews(
        reviews_path=reviews_path,
        output_dir=output_dir,
        model_name=args.model,
        max_reviews_per_item=args.max_reviews_per_item,
        max_text_len=args.max_text_len,
        max_items=args.max_items,
        min_reviews_per_item=args.min_reviews_per_item,
    )


if __name__ == "__main__":
    main()
