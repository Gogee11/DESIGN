import argparse
import datetime
import json
import os
import sys

import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config.configurator import configs
from load_data.build_data_handler import build_data_handler
from models.bulid_model import build_model
from trainer.utils import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Cold-start bucket evaluation for recommendation models")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to the target model checkpoint",
    )
    parser.add_argument(
        "--bucket_mode",
        type=str,
        choices=["ranges", "threshold"],
        default="ranges",
        help="Bucket mode: fixed ranges or cold/normal threshold",
    )
    parser.add_argument(
        "--bucket_ranges",
        type=str,
        default="1-3,4-7,8-15,16+",
        help="Range buckets for train interaction count, e.g. 1-3,4-7,8-15,16+",
    )
    parser.add_argument(
        "--cold_threshold",
        type=int,
        default=5,
        help="Cold-start threshold when bucket_mode=threshold",
    )
    parser.add_argument(
        "--eval_split",
        type=str,
        choices=["test", "val"],
        default="test",
        help="Evaluate on test or validation split",
    )
    parser.add_argument(
        "--eval_tag",
        type=str,
        default=None,
        help="Optional tag for this cold-start evaluation run",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
        help="Random seed for reproducibility",
    )
    args, _ = parser.parse_known_args()
    return args


def parse_bucket_ranges(bucket_ranges_text):
    buckets = []
    for raw_part in bucket_ranges_text.split(","):
        part = raw_part.strip()
        if not part:
            continue
        if part.endswith("+"):
            low = int(part[:-1])
            buckets.append({"name": part, "low": low, "high": None})
        else:
            low_text, high_text = part.split("-", 1)
            buckets.append({"name": part, "low": int(low_text), "high": int(high_text)})
    return buckets


def build_threshold_buckets(threshold):
    return [
        {"name": f"cold<= {threshold}", "low": 0, "high": threshold},
        {"name": f"normal> {threshold}", "low": threshold + 1, "high": None},
    ]


def bucket_name_for_count(count, buckets):
    for bucket in buckets:
        low = bucket["low"]
        high = bucket["high"]
        if high is None and count >= low:
            return bucket["name"]
        if high is not None and low <= count <= high:
            return bucket["name"]
    return "unbucketed"


def recall_at_k(ground_truth, ranked_items, k):
    if not ground_truth:
        return 0.0
    hits = sum(1 for item in ranked_items[:k] if item in ground_truth)
    return hits / len(ground_truth)


def ndcg_at_k(ground_truth, ranked_items, k):
    if not ground_truth:
        return 0.0
    dcg = 0.0
    for rank, item in enumerate(ranked_items[:k], start=1):
        if item in ground_truth:
            dcg += 1.0 / np.log2(rank + 1)
    ideal_hits = min(len(ground_truth), k)
    idcg = sum(1.0 / np.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def create_output_paths(model_name, dataset_name, checkpoint_path, split_name, eval_tag):
    output_dir = os.path.join("log", "cold_start", model_name)
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_stem = os.path.splitext(os.path.basename(checkpoint_path))[0]
    file_stem = f"{dataset_name}-{split_name}-{checkpoint_stem}-{timestamp}"
    if eval_tag:
        file_stem = f"{file_stem}-{eval_tag.strip()}"
    return (
        os.path.join(output_dir, f"{file_stem}.log"),
        os.path.join(output_dir, f"{file_stem}.json"),
    )


def select_dataloader(data_handler, split_name):
    return data_handler.test_dataloader if split_name == "test" else data_handler.valid_dataloader


def main():
    args = parse_args()
    set_seed(args.seed)

    if not os.path.exists(args.checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint_path}")

    buckets = (
        build_threshold_buckets(args.cold_threshold)
        if args.bucket_mode == "threshold"
        else parse_bucket_ranges(args.bucket_ranges)
    )

    data_handler = build_data_handler()
    data_handler.load_data()

    model = build_model(data_handler).to(configs["device"])
    state_dict = torch.load(args.checkpoint_path, map_location=configs["device"])
    model.load_state_dict(state_dict)
    model.eval()

    eval_dataloader = select_dataloader(data_handler, args.eval_split)
    train_csr = data_handler.trn_mat.tocsr()
    user_train_counts = np.asarray(train_csr.sum(axis=1)).reshape(-1).astype(np.int64)

    log_path, json_path = create_output_paths(
        configs["model"]["name"], configs["data"]["name"], args.checkpoint_path, args.eval_split, args.eval_tag
    )
    lines = []

    def emit(message=""):
        print(message)
        lines.append(message)

    emit("==== Cold-start Evaluation ====")
    emit(f"Model      : {configs['model']['name']}")
    emit(f"Dataset    : {configs['data']['name']}")
    emit(f"Split      : {args.eval_split}")
    emit(f"Checkpoint : {args.checkpoint_path}")
    emit(f"Log        : {log_path}")
    emit(f"JSON       : {json_path}")
    emit(f"Buckets    : {[bucket['name'] for bucket in buckets]}")

    bucket_results = {
        bucket["name"]: {
            "user_count": 0,
            "train_interaction_counts": [],
            "metrics": {
                "recall": np.zeros(len(configs["test"]["k"]), dtype=np.float64),
                "ndcg": np.zeros(len(configs["test"]["k"]), dtype=np.float64),
            },
        }
        for bucket in buckets
    }
    bucket_results["unbucketed"] = {
        "user_count": 0,
        "train_interaction_counts": [],
        "metrics": {
            "recall": np.zeros(len(configs["test"]["k"]), dtype=np.float64),
            "ndcg": np.zeros(len(configs["test"]["k"]), dtype=np.float64),
        },
    }

    topk_max = max(configs["test"]["k"])

    for _, tem in enumerate(eval_dataloader):
        if not isinstance(tem, list):
            tem = [tem]
        test_users = tem[0].numpy().tolist()
        batch_data = [tensor.long().to(configs["device"]) for tensor in tem]
        with torch.no_grad():
            batch_pred = model.full_predict(batch_data)
        batch_pred = batch_pred.clone()
        for i, user_idx in enumerate(test_users):
            history_items = eval_dataloader.dataset.csrmat[user_idx].indices.tolist()
            batch_pred[i, history_items] = -1e8
        _, batch_topk = torch.topk(batch_pred, k=topk_max)
        batch_topk = batch_topk.cpu().numpy()

        for i, user_idx in enumerate(test_users):
            train_count = int(user_train_counts[user_idx])
            bucket_name = bucket_name_for_count(train_count, buckets)
            ground_truth = list(eval_dataloader.dataset.user_pos_lists[user_idx])
            ranked_items = batch_topk[i].tolist()

            bucket_entry = bucket_results[bucket_name]
            bucket_entry["user_count"] += 1
            bucket_entry["train_interaction_counts"].append(train_count)

            for metric_index, k in enumerate(configs["test"]["k"]):
                bucket_entry["metrics"]["recall"][metric_index] += recall_at_k(ground_truth, ranked_items, k)
                bucket_entry["metrics"]["ndcg"][metric_index] += ndcg_at_k(ground_truth, ranked_items, k)

    summary = {
        "run_timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": configs["model"]["name"],
        "dataset": configs["data"]["name"],
        "split": args.eval_split,
        "checkpoint_path": args.checkpoint_path,
        "bucket_mode": args.bucket_mode,
        "bucket_ranges": args.bucket_ranges,
        "cold_threshold": args.cold_threshold,
        "k": configs["test"]["k"],
        "buckets": {},
    }

    emit("\n==== Bucket Results ====")
    for bucket_name, bucket_entry in bucket_results.items():
        user_count = bucket_entry["user_count"]
        if user_count == 0:
            continue
        recall_values = (bucket_entry["metrics"]["recall"] / user_count).tolist()
        ndcg_values = (bucket_entry["metrics"]["ndcg"] / user_count).tolist()
        avg_train_count = float(np.mean(bucket_entry["train_interaction_counts"]))
        summary["buckets"][bucket_name] = {
            "user_count": user_count,
            "avg_train_interactions": avg_train_count,
            "recall": recall_values,
            "ndcg": ndcg_values,
        }
        emit(f"[{bucket_name}] users={user_count} avg_train_interactions={avg_train_count:.2f}")
        emit("  " + " ".join(f"recall@{k}: {value:.4f}" for k, value in zip(configs["test"]["k"], recall_values)))
        emit("  " + " ".join(f"ndcg@{k}: {value:.4f}" for k, value in zip(configs["test"]["k"], ndcg_values)))

    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
