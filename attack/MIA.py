import argparse
import inspect
import os

import numpy as np
import torch
from scipy.sparse import coo_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from config.configurator import configs
from load_data.build_data_handler import build_data_handler
from models.bulid_model import build_model
from trainer.utils import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Membership Inference Attack for recommendation models")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Path to the target model checkpoint. Defaults to checkpoint/{model}/{model}-{dataset}-{seed}.pth",
    )
    parser.add_argument(
        "--member_limit",
        type=int,
        default=50000,
        help="Maximum number of member interactions used by the attack",
    )
    parser.add_argument(
        "--nonmember_limit",
        type=int,
        default=50000,
        help="Maximum number of non-member interactions used by the attack",
    )
    parser.add_argument(
        "--nonmember_source",
        type=str,
        choices=["heldout", "random", "mixed"],
        default="mixed",
        help="Source of non-member interactions",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.3,
        help="Attack train/test split ratio",
    )
    parser.add_argument(
        "--attack_seed",
        type=int,
        default=42,
        help="Random seed used by the attack pipeline",
    )
    args, _ = parser.parse_known_args()
    return args


def default_checkpoint_path():
    model_name = configs["model"]["name"]
    dataset_name = configs["data"]["name"]
    seed = configs["train"]["seed"]
    return os.path.join(
        "checkpoint",
        model_name,
        f"{model_name}-{dataset_name}-{seed}.pth",
    )


def ensure_coo(mat):
    if isinstance(mat, coo_matrix):
        return mat
    return mat.tocoo()


def sample_edges(rows, cols, sample_size, rng):
    total = len(rows)
    if total == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    if sample_size is None or sample_size >= total:
        idx = np.arange(total)
    else:
        idx = rng.choice(total, size=sample_size, replace=False)
    return rows[idx].astype(np.int64), cols[idx].astype(np.int64)


def sample_random_nonmember_edges(train_mat, sample_size, rng):
    train_dok = train_mat.todok()
    user_num, item_num = train_mat.shape

    sampled_users = []
    sampled_items = []
    seen = set()

    while len(sampled_users) < sample_size:
        user = int(rng.integers(user_num))
        item = int(rng.integers(item_num))
        key = (user, item)
        if key in train_dok or key in seen:
            continue
        seen.add(key)
        sampled_users.append(user)
        sampled_items.append(item)

    return np.asarray(sampled_users, dtype=np.int64), np.asarray(sampled_items, dtype=np.int64)


def build_attack_edges(data_handler, args):
    rng = np.random.default_rng(args.attack_seed)

    trn_mat = ensure_coo(data_handler.trn_mat)

    # Validation/test dataloaders only keep masks; reload matrices from handler for accurate held-out positives.
    val_file = ensure_coo(data_handler._load_one_mat(data_handler.val_file))
    tst_file = ensure_coo(data_handler._load_one_mat(data_handler.tst_file))

    member_users, member_items = sample_edges(trn_mat.row, trn_mat.col, args.member_limit, rng)

    nonmember_users_list = []
    nonmember_items_list = []

    if args.nonmember_source in {"heldout", "mixed"}:
        heldout_rows = np.concatenate([val_file.row, tst_file.row]).astype(np.int64)
        heldout_cols = np.concatenate([val_file.col, tst_file.col]).astype(np.int64)
        heldout_limit = args.nonmember_limit if args.nonmember_source == "heldout" else min(len(heldout_rows), args.nonmember_limit)
        h_users, h_items = sample_edges(heldout_rows, heldout_cols, heldout_limit, rng)
        nonmember_users_list.append(h_users)
        nonmember_items_list.append(h_items)

    current_nonmember_num = sum(len(x) for x in nonmember_users_list)
    if args.nonmember_source in {"random", "mixed"} and current_nonmember_num < args.nonmember_limit:
        random_needed = args.nonmember_limit - current_nonmember_num
        r_users, r_items = sample_random_nonmember_edges(trn_mat, random_needed, rng)
        nonmember_users_list.append(r_users)
        nonmember_items_list.append(r_items)

    if not nonmember_users_list:
        raise ValueError("Failed to construct non-member interactions.")

    nonmember_users = np.concatenate(nonmember_users_list)
    nonmember_items = np.concatenate(nonmember_items_list)

    return (member_users, member_items), (nonmember_users, nonmember_items)


def get_inference_embeddings(model):
    model.eval()
    if hasattr(model, "is_training"):
        model.is_training = False
    if hasattr(model, "final_embeds"):
        model.final_embeds = None

    with torch.no_grad():
        base_adj = getattr(model, "adj", None)
        attack_adj = base_adj
        if hasattr(model, "learn_graph_structure"):
            try:
                if configs["model"]["name"] == "bigcf_agr":
                    g_indices, g_values = model.learn_graph_structure(configs.get("cf_index", None))
                    result = model.forward(g_indices, g_values)
                    if isinstance(result, tuple) and len(result) >= 2:
                        return result[0], result[1]
                elif base_adj is not None:
                    attack_adj = model.learn_graph_structure(base_adj, configs.get("cf_index", None))
            except Exception:
                attack_adj = base_adj

        signature = inspect.signature(model.forward)
        param_names = list(signature.parameters.keys())[1:]

        def call_forward(adj_value):
            kwargs = {}
            if adj_value is not None and param_names:
                first_param = param_names[0]
                kwargs[first_param] = adj_value
            if "emb_type" in signature.parameters:
                kwargs["emb_type"] = "cf"
            if "keep_rate" in signature.parameters:
                kwargs["keep_rate"] = 1.0
            if "perturb" in signature.parameters:
                kwargs["perturb"] = False
            return model.forward(**kwargs)

        call_candidates = []
        if attack_adj is not None:
            call_candidates.append(lambda: call_forward(attack_adj))
        if base_adj is not None and attack_adj is not base_adj:
            call_candidates.append(lambda: call_forward(base_adj))
        call_candidates.append(lambda: call_forward(None))

        for caller in call_candidates:
            try:
                result = caller()
            except (TypeError, ValueError):
                continue
            if isinstance(result, tuple) and len(result) >= 2:
                return result[0], result[1]

    raise RuntimeError(f"Unable to infer embeddings from {configs['model']['name']}. forward signature: {signature}")


def build_attack_features(user_embeds, item_embeds, edge_users, edge_items, train_mat):
    device = user_embeds.device
    batch_size = 8192

    train_csr = train_mat.tocsr()
    train_csc = train_mat.tocsc()
    user_degree = np.asarray(train_csr.sum(axis=1)).reshape(-1).astype(np.float32)
    item_degree = np.asarray(train_csc.sum(axis=0)).reshape(-1).astype(np.float32)

    feature_blocks = []

    for start in range(0, len(edge_users), batch_size):
        end = min(start + batch_size, len(edge_users))
        batch_users = torch.from_numpy(edge_users[start:end]).to(device)
        batch_items = torch.from_numpy(edge_items[start:end]).to(device)

        u_emb = user_embeds[batch_users]
        i_emb = item_embeds[batch_items]

        score = torch.sum(u_emb * i_emb, dim=1)
        prob = torch.sigmoid(score)
        u_norm = torch.norm(u_emb, dim=1)
        i_norm = torch.norm(i_emb, dim=1)
        cosine = score / (u_norm * i_norm + 1e-12)
        l2_dist = torch.norm(u_emb - i_emb, dim=1)

        features = np.stack(
            [
                score.detach().cpu().numpy(),
                prob.detach().cpu().numpy(),
                cosine.detach().cpu().numpy(),
                u_norm.detach().cpu().numpy(),
                i_norm.detach().cpu().numpy(),
                l2_dist.detach().cpu().numpy(),
                user_degree[edge_users[start:end]],
                item_degree[edge_items[start:end]],
                user_degree[edge_users[start:end]] / max(train_mat.shape[1], 1),
                item_degree[edge_items[start:end]] / max(train_mat.shape[0], 1),
            ],
            axis=1,
        ).astype(np.float32)
        feature_blocks.append(features)

    return np.concatenate(feature_blocks, axis=0)


def train_attack_model(features, labels, test_size, seed):
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=test_size,
        random_state=seed,
        stratify=labels,
    )

    classifier = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed),
    )
    classifier.fit(x_train, y_train)

    y_prob = classifier.predict_proba(x_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(np.int64)

    return {
        "auc": roc_auc_score(y_test, y_prob),
        "acc": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "train_size": len(x_train),
        "test_size": len(x_test),
    }


def describe_privacy_risk(auc):
    if auc >= 0.90:
        return "Very high privacy leakage risk"
    if auc >= 0.80:
        return "High privacy leakage risk"
    if auc >= 0.65:
        return "Moderate privacy leakage risk"
    return "Low privacy leakage risk"


def main():
    args = parse_args()
    set_seed(args.attack_seed)

    print("==== Membership Inference Attack ====")
    print(f"Model   : {configs['model']['name']}")
    print(f"Dataset : {configs['data']['name']}")
    print(f"Device  : {configs['device']}")

    data_handler = build_data_handler()
    data_handler.load_data()

    model = build_model(data_handler).to(configs["device"])

    checkpoint_path = args.checkpoint_path or default_checkpoint_path()
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    state_dict = torch.load(checkpoint_path, map_location=configs["device"])
    model.load_state_dict(state_dict)
    print(f"Checkpoint: {checkpoint_path}")

    (member_users, member_items), (nonmember_users, nonmember_items) = build_attack_edges(data_handler, args)
    print(f"Members     : {len(member_users)}")
    print(f"Non-members : {len(nonmember_users)} ({args.nonmember_source})")

    user_embeds, item_embeds = get_inference_embeddings(model)

    member_features = build_attack_features(user_embeds, item_embeds, member_users, member_items, data_handler.trn_mat)
    nonmember_features = build_attack_features(user_embeds, item_embeds, nonmember_users, nonmember_items, data_handler.trn_mat)

    features = np.concatenate([member_features, nonmember_features], axis=0)
    labels = np.concatenate(
        [
            np.ones(len(member_features), dtype=np.int64),
            np.zeros(len(nonmember_features), dtype=np.int64),
        ],
        axis=0,
    )

    metrics = train_attack_model(features, labels, args.test_size, args.attack_seed)

    print("\n==== Attack Result ====")
    print(f"AUC       : {metrics['auc']:.4f}")
    print(f"ACC       : {metrics['acc']:.4f}")
    print(f"Precision : {metrics['precision']:.4f}")
    print(f"Recall    : {metrics['recall']:.4f}")
    print(f"F1        : {metrics['f1']:.4f}")
    print(f"Train/Test: {metrics['train_size']}/{metrics['test_size']}")

    print("\n==== Privacy Risk Interpretation ====")
    print(describe_privacy_risk(metrics["auc"]))


if __name__ == "__main__":
    main()
