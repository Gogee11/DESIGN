import argparse
import glob
import json
import os
import pickle

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


def resolve_default_meta_file(dataset_dir: str) -> str:
    candidates = [
        os.path.join(dataset_dir, "meta_clean.jsonl"),
        os.path.join(dataset_dir, "meta.jsonl"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path

    matched = sorted(glob.glob(os.path.join(dataset_dir, "meta*.jsonl")))
    if matched:
        return matched[0]

    raise FileNotFoundError(
        f"Cannot find meta file under {dataset_dir}. "
        "Please pass --meta_file explicitly."
    )


def generate_item_embeddings(
    meta_file: str,
    output_dir: str,
    model_name: str,
    max_items: int | None = None,
):
    print(f"Loading model: {model_name}")
    model = SentenceTransformer(model_name)

    print(f"Reading meta file: {meta_file}")
    items = []
    texts = []

    with open(meta_file, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Parse meta"):
            try:
                item = json.loads(line.strip())
            except Exception:
                continue

            asin = item.get("parent_asin") or item.get("asin") or ""
            title = item.get("title") or ""

            text_parts = [title]

            subtitle = item.get("subtitle")
            if subtitle:
                text_parts.append(subtitle)

            features = item.get("features")
            if isinstance(features, list):
                text_parts.extend(features)

            desc = item.get("description")
            if isinstance(desc, list):
                text_parts.extend(desc)
            elif isinstance(desc, str):
                text_parts.append(desc)

            categories = item.get("categories")
            if isinstance(categories, list):
                flat = []
                for cat in categories:
                    if isinstance(cat, list):
                        flat.extend(cat)
                    elif cat:
                        flat.append(str(cat))
                if flat:
                    text_parts.append(" ".join(flat))

            main_category = item.get("main_category")
            if main_category:
                text_parts.append(main_category)

            text = " ".join(str(p) for p in text_parts if p).strip()
            if text and asin:
                items.append(asin)
                texts.append(text)

    if max_items is not None and len(items) > max_items:
        idx = np.random.permutation(len(items))[:max_items]
        items = [items[i] for i in idx]
        texts = [texts[i] for i in idx]

    if not items:
        raise RuntimeError(f"No valid items parsed from {meta_file}")

    print(f"Encoding {len(items)} items ...")
    batch_size = 64
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Encode items"):
        batch = texts[i : i + batch_size]
        emb = model.encode(batch, normalize_embeddings=True)
        embeddings.append(emb)
    embeddings = np.vstack(embeddings).astype(np.float32)

    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "itm_emb_np.pkl")
    id_file = os.path.join(output_dir, "item_ids.pkl")

    with open(output_file, "wb") as f:
        pickle.dump(embeddings, f)
    with open(id_file, "wb") as f:
        pickle.dump(items, f)

    print(f"Saved embeddings: {output_file} {embeddings.shape}")
    print(f"Saved item ids:   {id_file} ({len(items)})")


def main():
    parser = argparse.ArgumentParser(description="Generate item embeddings from meta file.")
    parser.add_argument("--dataset", type=str, default="amazon")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--meta_file", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--model", type=str, default="D:/models/all-MiniLM-L6-v2")
    parser.add_argument("--max_items", type=int, default=None)
    args = parser.parse_args()

    dataset_dir = os.path.join(args.data_dir, args.dataset)
    meta_file = args.meta_file or resolve_default_meta_file(dataset_dir)
    output_dir = args.output_dir or dataset_dir

    generate_item_embeddings(
        meta_file=meta_file,
        output_dir=output_dir,
        model_name=args.model,
        max_items=args.max_items,
    )


if __name__ == "__main__":
    main()
