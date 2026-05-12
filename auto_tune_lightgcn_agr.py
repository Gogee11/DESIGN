import argparse
import datetime
import glob
import json
import os
import subprocess
import sys
from copy import deepcopy


DEFAULT_STAGEWISE_STAGES = [
    ("reg_weight", [1e-7, 5e-7, 1e-6, 2e-6, 5e-6, 1e-5]),
    ("keep_rate", [0.7, 0.8, 0.9]),
    ("beta", [5.0, 7.5, 10.0]),
    ("prf_weight", [0.05, 0.08, 0.1]),
    ("str_weight", [0.8, 1.0, 1.3]),
    ("alpha", [0.2, 0.25, 0.3]),
]


DEFAULT_GROUPED_EXPERIMENTS = [
    ("baseline", {}),
    (
        "train_control",
        {
            "train_batch_size": 1024,
            "train_patience": 1,
            "train_test_step": 1,
            "keep_rate": 0.7,
            "beta": 10.0,
            "reg_weight": 5e-6,
        },
    ),
    (
        "distill",
        {
            "prf_weight": 0.1,
            "str_weight": 1.3,
            "alpha": 0.3,
        },
    ),
    (
        "train_control_distill",
        {
            "train_batch_size": 1024,
            "train_patience": 1,
            "train_test_step": 1,
            "keep_rate": 0.7,
            "beta": 10.0,
            "reg_weight": 5e-6,
            "prf_weight": 0.1,
            "str_weight": 1.3,
            "alpha": 0.3,
        },
    ),
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Automatic experiments for lightgcn_agr privacy tuning"
    )
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    parser.add_argument("--device", type=str, default="cuda", help="cpu or cuda")
    parser.add_argument("--cuda", type=str, default="0", help="CUDA device id")
    parser.add_argument("--seed", type=int, default=2026, help="Training seed")
    parser.add_argument(
        "--mode",
        type=str,
        default="stagewise",
        choices=["grouped", "stagewise"],
        help="stagewise: greedy parameter search over sensitive parameters; grouped: run 4 defense groups",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=None,
        help="Optional fixed training batch size override; ignored if a candidate/group sets train_batch_size",
    )
    parser.add_argument("--train_epoch", type=int, default=200, help="Optional epoch override")
    parser.add_argument("--train_patience", type=int, default=None, help="Optional fixed patience override")
    parser.add_argument("--train_test_step", type=int, default=None, help="Optional fixed test_step override")
    parser.add_argument("--test_batch_size", type=int, default=None, help="Optional test batch size override")
    parser.add_argument("--member_limit", type=int, default=50000, help="MIA member sample limit")
    parser.add_argument("--nonmember_limit", type=int, default=50000, help="MIA non-member sample limit")
    parser.add_argument(
        "--nonmember_source",
        type=str,
        default="mixed",
        choices=["heldout", "random", "mixed"],
    )
    parser.add_argument("--test_size", type=float, default=0.3, help="MIA test split size")
    parser.add_argument("--attack_seed", type=int, default=42, help="MIA seed")
    parser.add_argument(
        "--privacy_weight",
        type=float,
        default=0.25,
        help="Penalty weight for MIA AUC in selection score",
    )
    parser.add_argument(
        "--utility_metric",
        type=str,
        default="recall",
        choices=["recall", "ndcg"],
        help="Utility metric for selection",
    )
    parser.add_argument(
        "--utility_k_index",
        type=int,
        default=-1,
        help="Index in metric list, default -1 means largest K",
    )
    parser.add_argument("--tag_prefix", type=str, default="autotune", help="Prefix for exp_tag/attack_tag")
    parser.add_argument("--resume_existing", action="store_true", help="Reuse finished experiments with same tag if found")
    parser.add_argument(
        "--continue_on_failure",
        action="store_true",
        help="Record failed candidates and continue instead of aborting the whole run",
    )
    return parser.parse_args()


def sanitize_value(value):
    if isinstance(value, float):
        return str(value).replace("-", "m").replace(".", "p")
    return str(value)


def checkpoint_dir(model_name):
    return os.path.join("checkpoint", model_name)


def mia_log_dir(model_name):
    return os.path.join("log", "mia", model_name)


def find_single_file(pattern):
    matches = sorted(glob.glob(pattern), key=os.path.getmtime)
    return matches[-1] if matches else None


def find_matching_files(pattern):
    return sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)


def find_training_metadata(model_name, tag):
    return find_single_file(os.path.join(checkpoint_dir(model_name), f"*{tag}*.json"))


def find_checkpoint_from_metadata(meta_path):
    if not meta_path:
        return None
    return os.path.splitext(meta_path)[0] + ".pth"


def find_mia_result(model_name, tag):
    return find_single_file(os.path.join(mia_log_dir(model_name), f"*{tag}*.json"))


def normalize_path(path):
    return os.path.normcase(os.path.normpath(str(path or "")))


def float_equal(a, b, tol=1e-12):
    return abs(float(a) - float(b)) <= tol


def mia_result_matches(mia_result, checkpoint_path, args, tag):
    if not mia_result:
        return False
    if normalize_path(mia_result.get("checkpoint_path")) != normalize_path(checkpoint_path):
        return False
    if mia_result.get("model") != "lightgcn_agr":
        return False
    if mia_result.get("dataset") != args.dataset:
        return False

    attack_args = mia_result.get("attack_args") or {}
    if attack_args.get("member_limit") != args.member_limit:
        return False
    if attack_args.get("nonmember_limit") != args.nonmember_limit:
        return False
    if attack_args.get("nonmember_source") != args.nonmember_source:
        return False
    if not float_equal(attack_args.get("test_size"), args.test_size):
        return False
    if attack_args.get("attack_seed") != args.attack_seed:
        return False
    if attack_args.get("attack_tag") != tag:
        return False
    return True


def find_valid_mia_result(model_name, tag, checkpoint_path, args):
    pattern = os.path.join(mia_log_dir(model_name), f"*{tag}*.json")
    for candidate in find_matching_files(pattern):
        try:
            mia_result = load_json(candidate)
        except Exception:
            continue
        if mia_result_matches(mia_result, checkpoint_path, args, tag):
            return candidate
    return None


def run_command(command):
    print("\n[RUN]", " ".join(command))
    subprocess.run(command, check=True)


def build_training_command(args, params, tag):
    command = [
        sys.executable,
        "main.py",
        "--model", "lightgcn_agr",
        "--dataset", args.dataset,
        "--device", args.device,
        "--cuda", args.cuda,
        "--seed", str(args.seed),
        "--exp_tag", tag,
    ]
    if args.train_batch_size is not None and "train_batch_size" not in params:
        command.extend(["--train_batch_size", str(args.train_batch_size)])
    if args.train_epoch is not None:
        command.extend(["--train_epoch", str(args.train_epoch)])
    if args.train_patience is not None and "train_patience" not in params:
        command.extend(["--train_patience", str(args.train_patience)])
    if args.train_test_step is not None and "train_test_step" not in params:
        command.extend(["--train_test_step", str(args.train_test_step)])
    if args.test_batch_size is not None:
        command.extend(["--test_batch_size", str(args.test_batch_size)])
    for key, value in params.items():
        command.extend([f"--{key}", str(value)])
    return command


def build_mia_command(args, checkpoint_path, tag):
    return [
        sys.executable,
        "-m",
        "attack.MIA",
        "--model", "lightgcn_agr",
        "--dataset", args.dataset,
        "--device", args.device,
        "--cuda", args.cuda,
        "--seed", str(args.seed),
        "--checkpoint_path", checkpoint_path,
        "--member_limit", str(args.member_limit),
        "--nonmember_limit", str(args.nonmember_limit),
        "--nonmember_source", args.nonmember_source,
        "--test_size", str(args.test_size),
        "--attack_seed", str(args.attack_seed),
        "--attack_tag", tag,
    ]


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_metric_value(training_meta, metric_name, metric_index):
    test_result = training_meta.get("test_result") or {}
    metric_values = test_result.get(metric_name)
    if not metric_values:
        raise ValueError(f"Missing metric '{metric_name}' in training metadata: {training_meta}")
    index = metric_index
    if index < 0:
        index = len(metric_values) + index
    return float(metric_values[index])


def selection_score(utility_value, mia_auc, privacy_weight):
    return utility_value - privacy_weight * max(mia_auc - 0.5, 0.0)


def build_tag(prefix, dataset, label):
    return f"{prefix}_{dataset}_{label}"


def evaluate_run(args, params, label):
    tag = build_tag(args.tag_prefix, args.dataset, label)
    model_name = "lightgcn_agr"

    training_meta_path = find_training_metadata(model_name, tag) if args.resume_existing else None
    checkpoint_path = find_checkpoint_from_metadata(training_meta_path) if training_meta_path else None

    if not training_meta_path or not checkpoint_path or not os.path.exists(checkpoint_path):
        run_command(build_training_command(args, params, tag))
        training_meta_path = find_training_metadata(model_name, tag)
        checkpoint_path = find_checkpoint_from_metadata(training_meta_path)
        if not training_meta_path or not checkpoint_path or not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Failed to locate checkpoint/metadata for tag: {tag}")

    mia_result_path = find_valid_mia_result(model_name, tag, checkpoint_path, args) if args.resume_existing else None
    if not mia_result_path:
        run_command(build_mia_command(args, checkpoint_path, tag))
        mia_result_path = find_valid_mia_result(model_name, tag, checkpoint_path, args)
        if not mia_result_path:
            raise FileNotFoundError(f"Failed to locate MIA result JSON for tag: {tag}")

    training_meta = load_json(training_meta_path)
    mia_result = load_json(mia_result_path)

    utility_value = get_metric_value(training_meta, args.utility_metric, args.utility_k_index)
    recall20 = get_metric_value(training_meta, "recall", -1)
    ndcg20 = get_metric_value(training_meta, "ndcg", -1)
    mia_auc = float(mia_result["metrics"]["auc"])
    score = selection_score(utility_value, mia_auc, args.privacy_weight)

    return {
        "tag": tag,
        "label": label,
        "params": deepcopy(params),
        "checkpoint_path": checkpoint_path,
        "training_meta_path": training_meta_path,
        "mia_result_path": mia_result_path,
        "utility_metric": args.utility_metric,
        "utility_value": utility_value,
        "recall20": recall20,
        "ndcg20": ndcg20,
        "mia_auc": mia_auc,
        "score": score,
        "training_meta": training_meta,
        "mia_result": mia_result,
    }


def failure_record(label, params, exc):
    return {
        "label": label,
        "params": deepcopy(params),
        "status": "failed",
        "error": str(exc),
    }


def write_summary(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def run_grouped(args, summary_path, run_timestamp):
    results = []
    failures = []

    for group_name, params in DEFAULT_GROUPED_EXPERIMENTS:
        label = f"group_{group_name}"
        try:
            result = evaluate_run(args, params, label)
        except Exception as exc:
            if not args.continue_on_failure:
                raise
            result = failure_record(label, params, exc)
            failures.append(result)
            print(f"[FAIL] {label} | {exc}")
        else:
            print(
                f"[RESULT] {label} | "
                f"{args.utility_metric}={result['utility_value']:.4f} | "
                f"recall20={result['recall20']:.4f} | "
                f"ndcg20={result['ndcg20']:.4f} | "
                f"mia_auc={result['mia_auc']:.4f} | score={result['score']:.4f}"
            )
        results.append(result)
        write_summary(
            summary_path,
            {
                "run_timestamp": run_timestamp,
                "mode": args.mode,
                "dataset": args.dataset,
                "device": args.device,
                "cuda": args.cuda,
                "seed": args.seed,
                "privacy_weight": args.privacy_weight,
                "utility_metric": args.utility_metric,
                "utility_k_index": args.utility_k_index,
                "member_limit": args.member_limit,
                "nonmember_limit": args.nonmember_limit,
                "nonmember_source": args.nonmember_source,
                "test_size": args.test_size,
                "attack_seed": args.attack_seed,
                "tag_prefix": args.tag_prefix,
                "grouped_results": results,
                "failures": failures,
            },
        )

    valid_results = [item for item in results if item.get("status") != "failed"]
    best_result = max(valid_results, key=lambda item: item["score"]) if valid_results else None
    return results, best_result


def run_stagewise(args, summary_path, run_timestamp):
    current_params = {}
    completed_stages = []
    failures = []

    for stage_idx, (param_name, candidate_values) in enumerate(DEFAULT_STAGEWISE_STAGES, start=1):
        print(f"\n===== Stage {stage_idx}: {param_name} =====")
        stage_results = []

        for candidate_value in candidate_values:
            label = f"s{stage_idx:02d}_{param_name}_{sanitize_value(candidate_value)}"
            candidate_params = deepcopy(current_params)
            candidate_params[param_name] = candidate_value
            try:
                result = evaluate_run(args, candidate_params, label)
            except Exception as exc:
                if not args.continue_on_failure:
                    raise
                result = failure_record(label, candidate_params, exc)
                failures.append(result)
                print(f"[FAIL] {label} | {exc}")
            else:
                print(
                    f"[RESULT] {param_name}={candidate_value} | "
                    f"{args.utility_metric}={result['utility_value']:.4f} | "
                    f"recall20={result['recall20']:.4f} | "
                    f"ndcg20={result['ndcg20']:.4f} | "
                    f"mia_auc={result['mia_auc']:.4f} | score={result['score']:.4f}"
                )
            stage_results.append(result)

        valid_stage_results = [item for item in stage_results if item.get("status") != "failed"]
        if not valid_stage_results:
            raise RuntimeError(f"All candidates failed in stage {stage_idx}: {param_name}")

        best_result = max(valid_stage_results, key=lambda item: item["score"])
        current_params[param_name] = best_result["params"][param_name]
        completed_stages.append({
            "stage_index": stage_idx,
            "param_name": param_name,
            "candidate_values": candidate_values,
            "best_value": best_result["params"][param_name],
            "best_score": best_result["score"],
            "best_utility_value": best_result["utility_value"],
            "best_recall20": best_result["recall20"],
            "best_ndcg20": best_result["ndcg20"],
            "best_mia_auc": best_result["mia_auc"],
            "best_tag": best_result["tag"],
            "results": stage_results,
        })

        write_summary(
            summary_path,
            {
                "run_timestamp": run_timestamp,
                "mode": args.mode,
                "dataset": args.dataset,
                "device": args.device,
                "cuda": args.cuda,
                "seed": args.seed,
                "privacy_weight": args.privacy_weight,
                "utility_metric": args.utility_metric,
                "utility_k_index": args.utility_k_index,
                "member_limit": args.member_limit,
                "nonmember_limit": args.nonmember_limit,
                "nonmember_source": args.nonmember_source,
                "test_size": args.test_size,
                "attack_seed": args.attack_seed,
                "tag_prefix": args.tag_prefix,
                "current_best_params": current_params,
                "completed_stages": completed_stages,
                "failures": failures,
            },
        )

        print(
            f"[BEST] Stage {stage_idx} -> {param_name}={best_result['params'][param_name]} | "
            f"{args.utility_metric}={best_result['utility_value']:.4f} | "
            f"recall20={best_result['recall20']:.4f} | "
            f"ndcg20={best_result['ndcg20']:.4f} | "
            f"mia_auc={best_result['mia_auc']:.4f} | score={best_result['score']:.4f}"
        )

    final_best = completed_stages[-1]["results"]
    valid_final = [item for item in final_best if item.get("status") != "failed"]
    best_result = max(valid_final, key=lambda item: item["score"])
    return completed_stages, best_result, current_params


def main():
    args = parse_args()
    run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_dir = os.path.join("log", "auto_tune", "lightgcn_agr")
    os.makedirs(summary_dir, exist_ok=True)
    summary_path = os.path.join(summary_dir, f"{args.dataset}_{run_timestamp}_{args.tag_prefix}_{args.mode}.json")

    if args.mode == "grouped":
        results, best_result = run_grouped(args, summary_path, run_timestamp)
        print("\n===== Grouped Experiments Finished =====")
        if best_result:
            print("Best group :", best_result["label"])
            print("Best params:", best_result["params"])
        print("Summary    :", summary_path)
        return

    completed_stages, best_result, current_params = run_stagewise(args, summary_path, run_timestamp)
    print("\n===== Stagewise Auto Tune Finished =====")
    print("Best params:", current_params)
    print("Best tag   :", best_result["tag"])
    print("Summary    :", summary_path)


if __name__ == "__main__":
    main()
