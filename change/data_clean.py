#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import gzip
import json
import os
import shutil
from collections import Counter


def smart_open(path, mode="rt", encoding="utf-8"):
    if path.endswith(".gz"):
        if "b" in mode:
            return gzip.open(path, mode)
        return gzip.open(path, mode, encoding=encoding)
    else:
        if "b" in mode:
            return open(path, mode)
        return open(path, mode, encoding=encoding)


def iter_jsonl(path):
    with smart_open(path, "rt", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line), line
            except Exception as e:
                print(f"[WARN] JSON 解析失败，line={lineno}, err={e}")
                continue


def build_edge_file(review_path, edge_path, rating_threshold=3.0, item_field="parent_asin"):
    """
    第一步：
    从 review 中抽取轻量边文件，只保留：
    user_id \t item_id \t timestamp

    这里已经做 rating 过滤，但不改原始 review 文件。
    """
    print("\n[Step 1] 构建 edge 文件（含 rating 过滤）...")

    total = 0
    kept = 0
    skipped = 0

    with open(edge_path, "w", encoding="utf-8") as fout:
        for obj, _ in iter_jsonl(review_path):
            total += 1

            rating = obj.get("rating")
            user_id = obj.get("user_id")
            item_id = obj.get(item_field)
            timestamp = obj.get("timestamp", 0)

            if rating is None or user_id is None or item_id is None:
                skipped += 1
                continue

            try:
                rating = float(rating)
            except Exception:
                skipped += 1
                continue

            if rating < rating_threshold:
                continue

            fout.write(f"{user_id}\t{item_id}\t{timestamp}\n")
            kept += 1

            if total % 1_000_000 == 0:
                print(f"  processed={total:,}, kept={kept:,}")

    print(f"[Step 1] 完成：total={total:,}, kept={kept:,}, skipped={skipped:,}")
    return {
        "raw_reviews": total,
        "kept_after_rating_filter": kept,
        "skipped_missing_or_invalid": skipped,
    }


def count_edge_degrees(edge_path):
    """
    统计 edge 文件中的 user/item 度数
    """
    user_counter = Counter()
    item_counter = Counter()
    total = 0

    with open(edge_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 3:
                continue
            user_id, item_id, _ = parts
            user_counter[user_id] += 1
            item_counter[item_id] += 1
            total += 1

    return user_counter, item_counter, total


def k_core_once_edge(input_edge, output_edge, k=5):
    """
    对轻量 edge 文件做单轮 k-core
    """
    user_counter, item_counter, total = count_edge_degrees(input_edge)

    kept = 0
    removed = 0

    with open(input_edge, "r", encoding="utf-8") as fin, open(output_edge, "w", encoding="utf-8") as fout:
        for line in fin:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 3:
                removed += 1
                continue

            user_id, item_id, timestamp = parts

            if user_counter[user_id] >= k and item_counter[item_id] >= k:
                fout.write(f"{user_id}\t{item_id}\t{timestamp}\n")
                kept += 1
            else:
                removed += 1

    return {
        "input_edges": total,
        "kept_edges": kept,
        "removed_edges": removed,
        "valid_users_before_filter": sum(1 for _, c in user_counter.items() if c >= k),
        "valid_items_before_filter": sum(1 for _, c in item_counter.items() if c >= k),
    }


def iterative_k_core_edge(edge_path, final_edge_path, k, work_dir):
    """
    在 edge 文件上迭代 k-core，直到稳定

    print(f"\n[Step 2] 开始在 edge 文件上执行 {k}-core ...")

    round_id = 1
    current_input = edge_path
    history = []

    while True:
        current_output = os.path.join(work_dir, f"kcore_round_{round_id}.tsv")
        stats = k_core_once_edge(current_input, current_output, k)
        history.append({"round": round_id, **stats})

        print(
            f"  Round {round_id}: "
            f"input={stats['input_edges']:,}, "
            f"kept={stats['kept_edges']:,}, "
            f"removed={stats['removed_edges']:,}"
        )

        if stats["removed_edges"] == 0:
            shutil.move(current_output, final_edge_path)
            break

        if current_input != edge_path and os.path.exists(current_input):
            os.remove(current_input)

        current_input = current_output
        round_id += 1

    user_counter, item_counter, total = count_edge_degrees(final_edge_path)

    print(f"[Step 2] 完成：final_edges={total:,}, users={len(user_counter):,}, items={len(item_counter):,}")

    return {
        "rounds": history,
        "final_edges": total,
        "final_users": len(user_counter),
        "final_items": len(item_counter),
        "min_user_degree": min(user_counter.values()) if user_counter else 0,
        "min_item_degree": min(item_counter.values()) if item_counter else 0,
    }
    """
    max_rounds = 5
    print(f"\n[Step 2] 开始在 edge 文件上执行 {k}-core ...")
    print(f"  最多迭代轮数: {max_rounds}")

    round_id = 1
    current_input = edge_path
    history = []
    converged = False

    while round_id <= max_rounds:
        current_output = os.path.join(work_dir, f"kcore_round_{round_id}.tsv")
        stats = k_core_once_edge(current_input, current_output, k)
        history.append({"round": round_id, **stats})

        print(
            f"  Round {round_id}: "
            f"input={stats['input_edges']:,}, "
            f"kept={stats['kept_edges']:,}, "
            f"removed={stats['removed_edges']:,}"
        )

        if stats["removed_edges"] == 0:
            shutil.move(current_output, final_edge_path)
            converged = True
            print(f"  已在第 {round_id} 轮收敛")
            break

        if round_id == max_rounds:
            shutil.move(current_output, final_edge_path)
            print(f"  达到最大轮数 {max_rounds}，提前停止")
            break

        if current_input != edge_path and os.path.exists(current_input):
            os.remove(current_input)

        current_input = current_output
        round_id += 1

    user_counter, item_counter, total = count_edge_degrees(final_edge_path)

    print(f"[Step 2] 完成：final_edges={total:,}, users={len(user_counter):,}, items={len(item_counter):,}")
    print(f"  是否完全收敛: {converged}")

    return {
        "rounds": history,
        "final_edges": total,
        "final_users": len(user_counter),
        "final_items": len(item_counter),
        "min_user_degree": min(user_counter.values()) if user_counter else 0,
        "min_item_degree": min(item_counter.values()) if item_counter else 0,
        "converged": converged,
        "max_rounds": max_rounds,
    }


def load_valid_pairs_and_items(edge_path):
    """
    从最终 edge 文件中读取：
    1. 合法 (user_id, item_id) 对
    2. 合法 item 集合
    """
    valid_pairs = set()
    valid_items = set()

    with open(edge_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 3:
                continue
            user_id, item_id, _ = parts
            valid_pairs.add((user_id, item_id))
            valid_items.add(item_id)

    return valid_pairs, valid_items


def filter_reviews_by_edge(review_path, out_path, valid_pairs, rating_threshold=3.0, item_field="parent_asin"):
    """
    根据最终保留的 pair，回筛 review
    输出 reviews_clean.jsonl
    不改变原始 JSON 结构
    """
    print("\n[Step 3] 根据最终 edge 回筛 reviews ...")

    total = 0
    kept = 0
    skipped = 0

    with open(out_path, "w", encoding="utf-8") as fout:
        for obj, raw_line in iter_jsonl(review_path):
            total += 1

            rating = obj.get("rating")
            user_id = obj.get("user_id")
            item_id = obj.get(item_field)

            if rating is None or user_id is None or item_id is None:
                skipped += 1
                continue

            try:
                rating = float(rating)
            except Exception:
                skipped += 1
                continue

            if rating < rating_threshold:
                continue

            if (str(user_id), str(item_id)) in valid_pairs:
                fout.write(raw_line + "\n")
                kept += 1

            if total % 1_000_000 == 0:
                print(f"  processed={total:,}, kept={kept:,}")

    print(f"[Step 3] 完成：total={total:,}, kept={kept:,}, skipped={skipped:,}")

    return {
        "raw_reviews_rechecked": total,
        "kept_reviews": kept,
        "skipped_missing_or_invalid": skipped,
    }


def filter_meta(meta_path, out_path, valid_items, item_field="parent_asin"):
    """
    过滤 meta，只保留最终 item
    不改变原始 JSON 结构
    """
    print("\n[Step 4] 过滤 meta ...")

    total = 0
    kept = 0
    skipped = 0

    with open(out_path, "w", encoding="utf-8") as fout:
        for obj, raw_line in iter_jsonl(meta_path):
            total += 1

            item_id = obj.get(item_field)
            if item_id is None:
                skipped += 1
                continue

            if str(item_id) in valid_items:
                fout.write(raw_line + "\n")
                kept += 1

            if total % 1_000_000 == 0:
                print(f"  processed meta={total:,}, kept={kept:,}")

    print(f"[Step 4] 完成：total={total:,}, kept={kept:,}, skipped={skipped:,}")

    return {
        "raw_meta": total,
        "kept_meta": kept,
        "skipped_missing_item_id": skipped,
    }


def main():
    # ======================================
    # 直接在这里改参数
    # ======================================
    review_path = "/data/amazon/reviews.jsonl"
    meta_path = "/data/amazon/meta_Kindle_Store.jsonl"
    out_dir = "E:/clean_data"

    rating_threshold = 2

    k_core = 5
    item_field = "parent_asin"

    keep_intermediate = False
    # ======================================

    os.makedirs(out_dir, exist_ok=True)
    work_dir = os.path.join(out_dir, "_tmp")
    os.makedirs(work_dir, exist_ok=True)

    edge_path = os.path.join(out_dir, "interactions_rating_filtered.tsv")
    final_edge_path = os.path.join(out_dir, "interactions_5core.tsv")
    reviews_clean_path = os.path.join(out_dir, "reviews_clean.jsonl")
    meta_clean_path = os.path.join(out_dir, "meta_clean.jsonl")
    stats_path = os.path.join(out_dir, "stats.json")

    all_stats = {
        "config": {
            "review_path": review_path,
            "meta_path": meta_path,
            "out_dir": out_dir,
            "rating_threshold": rating_threshold,
            "k_core": k_core,
            "item_field": item_field,
            "preserve_original_json_structure": True,
            "fast_mode": "k-core on lightweight edge list",
        }
    }

    all_stats["step1_build_edge"] = build_edge_file(
        review_path=review_path,
        edge_path=edge_path,
        rating_threshold=rating_threshold,
        item_field=item_field,
    )

    all_stats["step2_kcore_on_edge"] = iterative_k_core_edge(
        edge_path=edge_path,
        final_edge_path=final_edge_path,
        k=k_core,
        work_dir=work_dir,
    )

    valid_pairs, valid_items = load_valid_pairs_and_items(final_edge_path)

    all_stats["step3_filter_reviews"] = filter_reviews_by_edge(
        review_path=review_path,
        out_path=reviews_clean_path,
        valid_pairs=valid_pairs,
        rating_threshold=rating_threshold,
        item_field=item_field,
    )

    all_stats["step4_filter_meta"] = filter_meta(
        meta_path=meta_path,
        out_path=meta_clean_path,
        valid_items=valid_items,
        item_field=item_field,
    )

    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(all_stats, f, ensure_ascii=False, indent=2)

    if not keep_intermediate:
        for p in [edge_path]:
            if os.path.exists(p):
                os.remove(p)

        if os.path.exists(work_dir):
            for name in os.listdir(work_dir):
                fp = os.path.join(work_dir, name)
                if os.path.isfile(fp):
                    os.remove(fp)
            os.rmdir(work_dir)

    print("\n全部完成。输出文件：")
    print(f"  reviews_clean.jsonl : {reviews_clean_path}")
    print(f"  meta_clean.jsonl    : {meta_clean_path}")
    print(f"  interactions_5core  : {final_edge_path}")
    print(f"  stats.json          : {stats_path}")


if __name__ == "__main__":
    main()