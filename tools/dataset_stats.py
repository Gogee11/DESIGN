import argparse
import os
import pickle

import numpy as np
from scipy.sparse import coo_matrix


def parse_args():
    parser = argparse.ArgumentParser(description="Print dataset statistics for recommendation splits")
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
    return parser.parse_args()


def load_sparse_matrix(path):
    with open(path, "rb") as f:
        matrix = (pickle.load(f) != 0).astype(np.float32)
    if not isinstance(matrix, coo_matrix):
        matrix = coo_matrix(matrix)
    return matrix


def density_str(interactions, user_num, item_num):
    total_pairs = max(user_num * item_num, 1)
    density = interactions / total_pairs
    return density, f"{density * 100:.6f}%"


def summarize_dataset(dataset_dir, dataset_name):
    split_paths = {
        "train": os.path.join(dataset_dir, "trn_mat.pkl"),
        "val": os.path.join(dataset_dir, "val_mat.pkl"),
        "test": os.path.join(dataset_dir, "tst_mat.pkl"),
    }

    missing = [name for name, path in split_paths.items() if not os.path.exists(path)]
    if missing:
        print(f"==== {dataset_name} ====")
        print(f"Missing split files: {missing}")
        print()
        return

    train_mat = load_sparse_matrix(split_paths["train"])
    val_mat = load_sparse_matrix(split_paths["val"])
    test_mat = load_sparse_matrix(split_paths["test"])

    user_num, item_num = train_mat.shape
    train_interactions = int(train_mat.nnz)
    val_interactions = int(val_mat.nnz)
    test_interactions = int(test_mat.nnz)
    total_interactions = train_interactions + val_interactions + test_interactions
    density, density_percent = density_str(total_interactions, user_num, item_num)

    print(f"==== {dataset_name} ====")
    print(f"#Users           : {user_num}")
    print(f"#Items           : {item_num}")
    print(f"#Train           : {train_interactions}")
    print(f"#Val             : {val_interactions}")
    print(f"#Test            : {test_interactions}")
    print(f"#Interactions    : {total_interactions}")
    print(f"Density          : {density:.8f} ({density_percent})")
    print()


def main():
    args = parse_args()

    if args.dataset:
        dataset_names = [args.dataset]
    else:
        dataset_names = [
            name for name in sorted(os.listdir(args.data_dir))
            if os.path.isdir(os.path.join(args.data_dir, name))
        ]

    for dataset_name in dataset_names:
        dataset_dir = os.path.join(args.data_dir, dataset_name)
        if not os.path.isdir(dataset_dir):
            print(f"Skip non-existing dataset directory: {dataset_dir}")
            continue
        summarize_dataset(dataset_dir, dataset_name)


if __name__ == "__main__":
    main()
