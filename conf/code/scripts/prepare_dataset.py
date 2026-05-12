"""End-to-end dataset preparation for the LLM-AGR project.

Steps:
    1. Load raw Amazon-style review JSONL (one record per line).
    2. Apply rating filter (rating >= min_rating treated as positive interaction).
    3. Iteratively apply k-core filtering on (user, item) edges.
    4. Re-index user/item IDs to integer indices in first-appearance order.
    5. Random per-user 6:2:2 train/val/test split.
    6. Build sparse interaction matrices and dump trn_mat / val_mat / tst_mat.
    7. Encode user-side text (concatenation of reviews) and item-side text
       (title + first-K reviews) via SentenceTransformer; output usr_emb_np.pkl
       and itm_emb_np.pkl in the SAME order as user_ids.pkl / item_ids.pkl.

Default sentence encoder: BGE-small-en-v1.5 (BAAI, open-source, 384-dim) -- a
Chinese-team-published model. Override via --model.

Example:
    python -m scripts.prepare_dataset \\
        --dataset digital_music \\
        --raw data/digital_music/Digital_Music_5.json \\
        --model /root/private_data/models/bge-small-en-v1.5
"""
import argparse
import json
import os
import pickle
from collections import Counter, defaultdict

import numpy as np
from scipy.sparse import coo_matrix


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', type=str, required=True,
                   help="Output dataset name; output dir will be data/<dataset>")
    p.add_argument('--raw', type=str, required=True,
                   help="Raw reviews JSONL/JSON path. Each line a review record.")
    p.add_argument('--out_dir', type=str, default=None,
                   help="Output directory; default ./data/<dataset>")
    p.add_argument('--min_rating', type=float, default=3.0,
                   help="Keep interactions with rating >= min_rating")
    p.add_argument('--k_core', type=int, default=5,
                   help="Iterative k-core filter on user/item degree")
    p.add_argument('--train_ratio', type=float, default=0.6)
    p.add_argument('--val_ratio', type=float, default=0.2)
    p.add_argument('--seed', type=int, default=2025)
    p.add_argument('--model', type=str,
                   default='/root/private_data/models/bge-small-en-v1.5',
                   help="SentenceTransformer model path (or HuggingFace name).")
    p.add_argument('--max_reviews_per_user', type=int, default=10)
    p.add_argument('--max_reviews_per_item', type=int, default=20)
    p.add_argument('--max_text_len', type=int, default=4000)
    p.add_argument('--encode_batch_size', type=int, default=64)
    p.add_argument('--skip_encoding', action='store_true',
                   help="Skip embedding step; only build matrices.")
    return p.parse_args()


def iter_reviews(path):
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def get_uid(r):
    return r.get('user_id') or r.get('reviewerID')


def get_iid(r):
    return r.get('parent_asin') or r.get('asin')


def get_rating(r):
    rv = r.get('rating', r.get('overall'))
    if rv is None:
        return None
    try:
        return float(rv)
    except Exception:
        return None


def get_text(r):
    title = r.get('title') or r.get('summary') or ''
    body = r.get('text') or r.get('reviewText') or ''
    return f"{title} {body}".strip()


def stage1_filter_and_kcore(args):
    print(f"[stage1] reading {args.raw}, rating>={args.min_rating}, k_core={args.k_core}")
    edges = []
    edge_text = []
    for r in iter_reviews(args.raw):
        uid, iid, rating = get_uid(r), get_iid(r), get_rating(r)
        if uid is None or iid is None or rating is None or rating < args.min_rating:
            continue
        edges.append((uid, iid))
        edge_text.append((uid, iid, get_text(r)))
    print(f"[stage1] after rating filter: {len(edges)} edges")

    # iterative k-core
    keep = set(range(len(edges)))
    while True:
        u_cnt = Counter(); i_cnt = Counter()
        for idx in keep:
            u, i = edges[idx]
            u_cnt[u] += 1
            i_cnt[i] += 1
        new_keep = {idx for idx in keep
                    if u_cnt[edges[idx][0]] >= args.k_core
                    and i_cnt[edges[idx][1]] >= args.k_core}
        if len(new_keep) == len(keep):
            break
        keep = new_keep
        print(f"[stage1] k-core iter: {len(keep)} edges, {len(u_cnt)} users, {len(i_cnt)} items")
    edges = [edges[i] for i in keep]
    edge_text = [edge_text[i] for i in keep]
    print(f"[stage1] final: {len(edges)} edges")
    return edges, edge_text


def stage2_index_and_split(edges, args):
    print(f"[stage2] indexing and splitting {args.train_ratio}/{args.val_ratio}/...")
    user_ids = []
    item_ids = []
    user2idx = {}
    item2idx = {}
    indexed = []
    for u, i in edges:
        if u not in user2idx:
            user2idx[u] = len(user_ids); user_ids.append(u)
        if i not in item2idx:
            item2idx[i] = len(item_ids); item_ids.append(i)
        indexed.append((user2idx[u], item2idx[i]))

    num_users = len(user_ids)
    num_items = len(item_ids)
    by_user = defaultdict(list)
    for u, i in indexed:
        by_user[u].append(i)

    rng = np.random.default_rng(args.seed)
    trn_r, trn_c, val_r, val_c, tst_r, tst_c = [], [], [], [], [], []
    e1 = args.train_ratio
    e2 = args.train_ratio + args.val_ratio
    for u in range(num_users):
        items_u = by_user.get(u, [])
        if not items_u:
            continue
        items_u = list(set(items_u))  # de-dup repeated interactions
        rng.shuffle(items_u)
        n = len(items_u)
        b1 = max(int(n * e1), 1) if n >= 3 else max(n - 2, 1)
        b2 = max(int(n * e2), b1 + 1) if n >= 3 else max(n - 1, b1)
        for j, it in enumerate(items_u):
            if j < b1:
                trn_r.append(u); trn_c.append(it)
            elif j < b2:
                val_r.append(u); val_c.append(it)
            else:
                tst_r.append(u); tst_c.append(it)

    trn_mat = coo_matrix(
        (np.ones(len(trn_r)), (trn_r, trn_c)), shape=(num_users, num_items))
    val_mat = coo_matrix(
        (np.ones(len(val_r)), (val_r, val_c)), shape=(num_users, num_items))
    tst_mat = coo_matrix(
        (np.ones(len(tst_r)), (tst_r, tst_c)), shape=(num_users, num_items))
    print(f"[stage2] users={num_users}, items={num_items}, "
          f"trn={trn_mat.nnz}, val={val_mat.nnz}, tst={tst_mat.nnz}")
    return user_ids, item_ids, user2idx, item2idx, trn_mat, val_mat, tst_mat


def stage3_collect_text(edge_text, user2idx, item2idx, args):
    print("[stage3] grouping review text by user/item")
    user_texts = defaultdict(list)
    item_texts = defaultdict(list)
    for u, i, txt in edge_text:
        if not txt:
            continue
        if u in user2idx and len(user_texts[user2idx[u]]) < args.max_reviews_per_user:
            user_texts[user2idx[u]].append(txt)
        if i in item2idx and len(item_texts[item2idx[i]]) < args.max_reviews_per_item:
            item_texts[item2idx[i]].append(txt)

    def join(texts, n):
        joined = ' '.join(texts[:n])
        return joined[:args.max_text_len] if len(joined) > args.max_text_len else joined

    user_inputs = [join(user_texts.get(u, []), args.max_reviews_per_user)
                   for u in range(len(user2idx))]
    item_inputs = [join(item_texts.get(i, []), args.max_reviews_per_item)
                   for i in range(len(item2idx))]
    user_inputs = [s if s else '<empty>' for s in user_inputs]
    item_inputs = [s if s else '<empty>' for s in item_inputs]
    return user_inputs, item_inputs


def stage4_encode(texts, args):
    from sentence_transformers import SentenceTransformer
    print(f"[stage4] loading encoder {args.model}")
    model = SentenceTransformer(args.model)
    bs = args.encode_batch_size
    embs = []
    from tqdm import tqdm
    for i in tqdm(range(0, len(texts), bs), desc='Encoding'):
        batch = texts[i:i + bs]
        emb = model.encode(batch, normalize_embeddings=True, show_progress_bar=False)
        embs.append(emb)
    return np.vstack(embs).astype(np.float32)


def main():
    args = parse_args()
    out_dir = args.out_dir or os.path.join('data', args.dataset)
    os.makedirs(out_dir, exist_ok=True)

    edges, edge_text = stage1_filter_and_kcore(args)
    user_ids, item_ids, user2idx, item2idx, trn_mat, val_mat, tst_mat = \
        stage2_index_and_split(edges, args)

    with open(os.path.join(out_dir, 'user_ids.pkl'), 'wb') as f:
        pickle.dump(user_ids, f)
    with open(os.path.join(out_dir, 'item_ids.pkl'), 'wb') as f:
        pickle.dump(item_ids, f)
    with open(os.path.join(out_dir, 'trn_mat.pkl'), 'wb') as f:
        pickle.dump(trn_mat, f)
    with open(os.path.join(out_dir, 'val_mat.pkl'), 'wb') as f:
        pickle.dump(val_mat, f)
    with open(os.path.join(out_dir, 'tst_mat.pkl'), 'wb') as f:
        pickle.dump(tst_mat, f)

    stats = {
        'num_users': len(user_ids),
        'num_items': len(item_ids),
        'trn_interactions': int(trn_mat.nnz),
        'val_interactions': int(val_mat.nnz),
        'tst_interactions': int(tst_mat.nnz),
        'min_rating': args.min_rating,
        'k_core': args.k_core,
        'train_ratio': args.train_ratio,
        'val_ratio': args.val_ratio,
        'seed': args.seed,
        'src': args.raw,
        'encoder_model': args.model,
    }
    with open(os.path.join(out_dir, 'data_stats.json'), 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"[done] wrote matrices to {out_dir}")

    if args.skip_encoding:
        print("[stage4] skipped (--skip_encoding)")
        return

    user_inputs, item_inputs = stage3_collect_text(edge_text, user2idx, item2idx, args)
    user_emb = stage4_encode(user_inputs, args)
    item_emb = stage4_encode(item_inputs, args)
    with open(os.path.join(out_dir, 'usr_emb_np.pkl'), 'wb') as f:
        pickle.dump(user_emb, f)
    with open(os.path.join(out_dir, 'itm_emb_np.pkl'), 'wb') as f:
        pickle.dump(item_emb, f)
    stats['embedding_dim'] = int(user_emb.shape[1])
    with open(os.path.join(out_dir, 'data_stats.json'), 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"[done] user_emb {user_emb.shape}, item_emb {item_emb.shape}")


if __name__ == '__main__':
    main()
