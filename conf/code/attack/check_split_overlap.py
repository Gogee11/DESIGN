import argparse
import os
import pickle

import numpy as np
from scipy.sparse import coo_matrix


def parse_args():
    parser = argparse.ArgumentParser(
        description="Check overlap leakage among train/val/test interaction splits"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Root data directory",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Dataset name. If omitted, scan all datasets under data_dir",
    )
    args, _ = parser.parse_known_args()
    return args


def load_sparse_matrix(path):
    with open(path, "rb") as file_obj:
        matrix = (pickle.load(file_obj) != 0).astype(np.float32)
    if not isinstance(matrix, coo_matrix):
        matrix = coo_matrix(matrix)
    return matrix


def matrix_to_edge_set(matrix):
    return set(zip(matrix.row.tolist(), matrix.col.tolist()))


def compute_user_overlap(left_edges, right_edges):
    left_users = {user for user, _ in left_edges}
    right_users = {user for user, _ in right_edges}
    overlap_users = left_users & right_users
    return len(left_users), len(right_users), len(overlap_users)


def summarize_pair(name_a, edges_a, name_b, edges_b):
    overlap_edges = edges_a & edges_b
    overlap_count = len(overlap_edges)
    denom = min(len(edges_a), len(edges_b)) if min(len(edges_a), len(edges_b)) > 0 else 1
    overlap_ratio = overlap_count / denom

    users_a, users_b, overlap_users = compute_user_overlap(edges_a, edges_b)

    print(f"{name_a} vs {name_b}")
    print(f"  Edge overlap count : {overlap_count}")
    print(f"  Edge overlap ratio : {overlap_ratio:.6f}")
    print(f"  User overlap count : {overlap_users}")
    print(f"  User counts        : {users_a} / {users_b}")

    if overlap_count > 0:
        preview = list(sorted(overlap_edges))[:10]
        print(f"  Sample overlapped edges: {preview}")
    print()


def check_dataset(dataset_dir, dataset_name):
    split_files = {
        "train": os.path.join(dataset_dir, "trn_mat.pkl"),
        "val": os.path.join(dataset_dir, "val_mat.pkl"),
        "test": os.path.join(dataset_dir, "tst_mat.pkl"),
    }

    missing = [name for name, path in split_files.items() if not os.path.exists(path)]
    if missing:
        print(f"==== {dataset_name} ====")
        print(f"Missing split files: {missing}")
        print()
        return

    matrices = {name: load_sparse_matrix(path) for name, path in split_files.items()}
    edge_sets = {name: matrix_to_edge_set(matrix) for name, matrix in matrices.items()}

    print(f"==== {dataset_name} ====")
    for split_name, edges in edge_sets.items():
        print(f"{split_name:<5} edges: {len(edges)}")
    print()

    summarize_pair("train", edge_sets["train"], "val", edge_sets["val"])
    summarize_pair("train", edge_sets["train"], "test", edge_sets["test"])
    summarize_pair("val", edge_sets["val"], "test", edge_sets["test"])


def main():
    args = parse_args()

    if args.dataset is not None:
        dataset_names = [args.dataset]
    else:
        dataset_names = [
            name
            for name in sorted(os.listdir(args.data_dir))
            if os.path.isdir(os.path.join(args.data_dir, name))
        ]

    for dataset_name in dataset_names:
        dataset_dir = os.path.join(args.data_dir, dataset_name)
        if not os.path.isdir(dataset_dir):
            print(f"Skip non-existing dataset directory: {dataset_dir}")
            continue
        check_dataset(dataset_dir, dataset_name)


if __name__ == "__main__":
    main()
