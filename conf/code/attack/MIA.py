"""Membership Inference Attack against trained recommender checkpoints.

Pipeline:
    1. Load a trained ckpt (LightGCN / LightGCN_AGR / etc).
    2. Compute user/item embeddings from the model's full inference forward.
    3. For each candidate (user, item) edge, build a hand-crafted feature vector
       from the embeddings (score, sigmoid prob, cosine, norms, l2 distance) plus
       degree-based features.
    4. Member edges = sampled training edges. Non-member edges = held-out + random.
    5. Train a logistic-regression classifier and report AUC / ACC / F1.
"""

import argparse
import inspect
import os
import json

import numpy as np
import torch
from scipy.sparse import coo_matrix

from config.configurator import configs
from load_data.build_data_handler import build_data_handler
from models.bulid_model import build_model
from trainer.utils import set_seed

# sklearn is optional. If available we use its LogisticRegression/MLP, otherwise
# we fall back to a pure-PyTorch logistic regression and pure-numpy metrics.
try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        accuracy_score, f1_score, precision_score, recall_score, roc_auc_score)
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    HAVE_SKLEARN = True
except ImportError:
    HAVE_SKLEARN = False


def _fallback_train_test_split(features, labels, test_size, seed):
    rng = np.random.default_rng(seed)
    n = len(features)
    perm = rng.permutation(n)
    n_test = int(round(test_size * n))
    te = perm[:n_test]; tr = perm[n_test:]
    return features[tr], features[te], labels[tr], labels[te]


def _fallback_metrics(y_true, y_prob, y_pred):
    y_true = np.asarray(y_true).astype(np.int64)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    y_pred = np.asarray(y_pred).astype(np.int64)
    # AUC via Mann-Whitney U statistic
    pos = y_prob[y_true == 1]; neg = y_prob[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        auc = 0.5
    else:
        order = np.argsort(y_prob)
        ranks = np.empty(len(y_prob), dtype=np.float64)
        ranks[order] = np.arange(1, len(y_prob) + 1)
        # ties: average rank
        from scipy.stats import rankdata
        ranks = rankdata(y_prob)
        auc = (ranks[y_true == 1].sum() - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg))
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    acc = (tp + tn) / max(len(y_true), 1)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    return float(auc), float(acc), float(prec), float(rec), float(f1)


def _torch_logistic(x_tr, y_tr, x_te, y_te, seed, epochs=300, lr=0.05, wd=1e-4):
    """Pure-PyTorch logistic regression with feature standardization."""
    torch.manual_seed(seed)
    mu = x_tr.mean(axis=0, keepdims=True)
    sigma = x_tr.std(axis=0, keepdims=True) + 1e-8
    Xt = torch.from_numpy(((x_tr - mu) / sigma).astype(np.float32))
    Yt = torch.from_numpy(y_tr.astype(np.float32))
    Xe = torch.from_numpy(((x_te - mu) / sigma).astype(np.float32))
    n_feat = Xt.shape[1]
    w = torch.zeros(n_feat, requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    # class-balance weights
    pos = (y_tr == 1).sum(); neg = len(y_tr) - pos
    w_pos = neg / max(pos + neg, 1) * 2.0
    w_neg = pos / max(pos + neg, 1) * 2.0
    cw = torch.from_numpy(np.where(y_tr == 1, w_pos, w_neg).astype(np.float32))
    opt = torch.optim.Adam([w, b], lr=lr, weight_decay=wd)
    for _ in range(epochs):
        opt.zero_grad()
        logit = Xt @ w + b
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logit, Yt, weight=cw)
        loss.backward(); opt.step()
    with torch.no_grad():
        y_prob = torch.sigmoid(Xe @ w + b).cpu().numpy()
    y_pred = (y_prob >= 0.5).astype(np.int64)
    return y_prob, y_pred


def parse_args():
    p = argparse.ArgumentParser(description="MIA against a recommender ckpt")
    p.add_argument('--checkpoint_path', type=str, default=None,
                   help="Default: checkpoint/{model}/{model}-{dataset}-{seed}.pth")
    p.add_argument('--member_limit', type=int, default=50000)
    p.add_argument('--nonmember_limit', type=int, default=50000)
    p.add_argument('--nonmember_source', type=str,
                   choices=['heldout', 'random', 'mixed'], default='mixed')
    p.add_argument('--test_size', type=float, default=0.3)
    p.add_argument('--attack_seed', type=int, default=42)
    p.add_argument('--attacker', type=str, choices=['lr', 'mlp'], default='lr',
                   help="Attack classifier: logistic regression (default) or MLP.")
    p.add_argument('--score_temperature', type=float, default=1.0,
                   help="Test-time defense: scale embeddings by 1/T before computing MIA features.")
    p.add_argument('--predict_noise_std', type=float, default=0.0,
                   help="Test-time defense: add Gaussian noise on inference embeddings.")
    p.add_argument('--balance_by_degree', type=int, default=0,
                   help="If 1, stratify member/non-member by user-degree quantile bins and "
                        "downsample to equalize, removing the degree-only data leakage floor.")
    p.add_argument('--num_degree_bins', type=int, default=5)
    p.add_argument('--out_json', type=str, default=None,
                   help="If set, append metrics as a JSON line to this file.")
    args, _ = p.parse_known_args()
    return args


def default_ckpt_path():
    return os.path.join(
        'checkpoint',
        configs['model']['name'],
        f"{configs['model']['name']}-{configs['data']['name']}-{configs['train']['seed']}.pth",
    )


def ensure_coo(mat):
    return mat if isinstance(mat, coo_matrix) else mat.tocoo()


def sample_edges(rows, cols, sample_size, rng):
    n = len(rows)
    if n == 0:
        return np.empty(0, np.int64), np.empty(0, np.int64)
    if sample_size is None or sample_size >= n:
        idx = np.arange(n)
    else:
        idx = rng.choice(n, size=sample_size, replace=False)
    return rows[idx].astype(np.int64), cols[idx].astype(np.int64)


def sample_random_nonmember(trn_mat, sample_size, rng):
    train_dok = trn_mat.todok()
    un, im = trn_mat.shape
    users, items = [], []
    seen = set()
    while len(users) < sample_size:
        u = int(rng.integers(un))
        i = int(rng.integers(im))
        key = (u, i)
        if key in train_dok or key in seen:
            continue
        seen.add(key)
        users.append(u)
        items.append(i)
    return np.asarray(users, np.int64), np.asarray(items, np.int64)


def build_attack_edges(data_handler, args):
    rng = np.random.default_rng(args.attack_seed)
    trn_mat = ensure_coo(data_handler.trn_mat)
    val_mat = ensure_coo(data_handler._load_one_mat(data_handler.val_file))
    tst_mat = ensure_coo(data_handler._load_one_mat(data_handler.tst_file))

    member_users, member_items = sample_edges(
        trn_mat.row, trn_mat.col, args.member_limit, rng)

    nm_users, nm_items = [], []
    if args.nonmember_source in ('heldout', 'mixed'):
        h_rows = np.concatenate([val_mat.row, tst_mat.row]).astype(np.int64)
        h_cols = np.concatenate([val_mat.col, tst_mat.col]).astype(np.int64)
        budget = args.nonmember_limit if args.nonmember_source == 'heldout' \
            else min(len(h_rows), args.nonmember_limit)
        u, i = sample_edges(h_rows, h_cols, budget, rng)
        nm_users.append(u); nm_items.append(i)

    have = sum(len(x) for x in nm_users)
    if args.nonmember_source in ('random', 'mixed') and have < args.nonmember_limit:
        need = args.nonmember_limit - have
        u, i = sample_random_nonmember(trn_mat, need, rng)
        nm_users.append(u); nm_items.append(i)

    if not nm_users:
        raise ValueError("No non-member edges built")
    return (member_users, member_items), (
        np.concatenate(nm_users), np.concatenate(nm_items))


def get_inference_embeddings(model):
    """Run model forward in eval mode to get final user/item embeddings."""
    model.eval()
    if hasattr(model, 'is_training'):
        model.is_training = False
    if hasattr(model, 'final_embeds'):
        model.final_embeds = None

    with torch.no_grad():
        sig = inspect.signature(model.forward)
        kwargs = {}
        if 'adj' in sig.parameters:
            kwargs['adj'] = getattr(model, 'adj', None)
        elif 'a' in sig.parameters:
            kwargs['a'] = getattr(model, 'adj', None)
        if 'keep_rate' in sig.parameters:
            kwargs['keep_rate'] = 1.0
        result = model.forward(**kwargs)

    if isinstance(result, tuple) and len(result) >= 2:
        return result[0], result[1]
    raise RuntimeError(
        f"forward did not return (user_embeds, item_embeds): signature={sig}")


def build_features(user_embeds, item_embeds, edge_users, edge_items, trn_mat):
    device = user_embeds.device
    bs = 8192
    train_csr = trn_mat.tocsr()
    train_csc = trn_mat.tocsc()
    user_deg = np.asarray(train_csr.sum(axis=1)).reshape(-1).astype(np.float32)
    item_deg = np.asarray(train_csc.sum(axis=0)).reshape(-1).astype(np.float32)
    blocks = []
    for s in range(0, len(edge_users), bs):
        e = min(s + bs, len(edge_users))
        u_idx = torch.from_numpy(edge_users[s:e]).to(device)
        i_idx = torch.from_numpy(edge_items[s:e]).to(device)
        u_e = user_embeds[u_idx]
        i_e = item_embeds[i_idx]
        score = torch.sum(u_e * i_e, dim=1)
        prob = torch.sigmoid(score)
        u_norm = torch.norm(u_e, dim=1)
        i_norm = torch.norm(i_e, dim=1)
        cos = score / (u_norm * i_norm + 1e-12)
        l2 = torch.norm(u_e - i_e, dim=1)
        feat = np.stack(
            [
                score.detach().cpu().numpy(),
                prob.detach().cpu().numpy(),
                cos.detach().cpu().numpy(),
                u_norm.detach().cpu().numpy(),
                i_norm.detach().cpu().numpy(),
                l2.detach().cpu().numpy(),
                user_deg[edge_users[s:e]],
                item_deg[edge_items[s:e]],
                user_deg[edge_users[s:e]] / max(trn_mat.shape[1], 1),
                item_deg[edge_items[s:e]] / max(trn_mat.shape[0], 1),
            ],
            axis=1,
        ).astype(np.float32)
        blocks.append(feat)
    return np.concatenate(blocks, axis=0)


def train_attacker(features, labels, args):
    if HAVE_SKLEARN:
        x_tr, x_te, y_tr, y_te = train_test_split(
            features, labels, test_size=args.test_size,
            random_state=args.attack_seed, stratify=labels)
        if args.attacker == 'lr':
            clf = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    max_iter=2000, class_weight='balanced',
                    random_state=args.attack_seed),
            )
        else:
            from sklearn.neural_network import MLPClassifier
            clf = make_pipeline(
                StandardScaler(),
                MLPClassifier(
                    hidden_layer_sizes=(64, 32), max_iter=300,
                    random_state=args.attack_seed),
            )
        clf.fit(x_tr, y_tr)
        y_prob = clf.predict_proba(x_te)[:, 1]
        y_pred = (y_prob >= 0.5).astype(np.int64)
        return {
            'auc': float(roc_auc_score(y_te, y_prob)),
            'acc': float(accuracy_score(y_te, y_pred)),
            'precision': float(precision_score(y_te, y_pred, zero_division=0)),
            'recall': float(recall_score(y_te, y_pred, zero_division=0)),
            'f1': float(f1_score(y_te, y_pred, zero_division=0)),
            'train_size': int(len(x_tr)),
            'test_size': int(len(x_te)),
            'attacker_backend': 'sklearn',
        }

    # sklearn unavailable: pure-PyTorch fallback
    print("[MIA] sklearn not available; using pure-PyTorch logistic regression attacker")
    x_tr, x_te, y_tr, y_te = _fallback_train_test_split(
        features, labels, args.test_size, args.attack_seed)
    y_prob, y_pred = _torch_logistic(x_tr, y_tr, x_te, y_te, args.attack_seed)
    auc, acc, prec, rec, f1 = _fallback_metrics(y_te, y_prob, y_pred)
    return {
        'auc': auc, 'acc': acc,
        'precision': prec, 'recall': rec, 'f1': f1,
        'train_size': int(len(x_tr)), 'test_size': int(len(x_te)),
        'attacker_backend': 'torch',
    }


def describe_risk(auc):
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
    print(f"Model    : {configs['model']['name']}")
    print(f"Dataset  : {configs['data']['name']}")
    print(f"Device   : {configs['device']}")
    print(f"Attacker : {args.attacker}")

    data_handler = build_data_handler()
    data_handler.load_data()
    model = build_model(data_handler).to(configs['device'])

    ckpt = args.checkpoint_path or default_ckpt_path()
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    state = torch.load(ckpt, map_location=configs['device'])
    model.load_state_dict(state)
    print(f"Checkpoint: {ckpt}")

    (mu, mi), (nu, ni) = build_attack_edges(data_handler, args)
    print(f"Members     : {len(mu)}")
    print(f"Non-members : {len(nu)} ({args.nonmember_source})")

    if args.balance_by_degree:
        from scipy.sparse import coo_matrix
        trn_csr = ensure_coo(data_handler.trn_mat).tocsr()
        user_deg = np.asarray(trn_csr.sum(axis=1)).reshape(-1)
        m_deg = user_deg[mu]
        n_deg = user_deg[nu]
        # bin by global quantiles of user_deg over members+nonmembers
        all_deg = np.concatenate([m_deg, n_deg])
        q = np.linspace(0, 100, args.num_degree_bins + 1)[1:-1]
        bins = np.percentile(all_deg, q) if len(q) > 0 else []
        m_bin = np.digitize(m_deg, bins)
        n_bin = np.digitize(n_deg, bins)
        rng = np.random.default_rng(args.attack_seed)
        keep_m, keep_n = [], []
        for b in range(args.num_degree_bins):
            mi_b = np.where(m_bin == b)[0]
            ni_b = np.where(n_bin == b)[0]
            k = min(len(mi_b), len(ni_b))
            if k == 0:
                continue
            keep_m.append(rng.choice(mi_b, size=k, replace=False))
            keep_n.append(rng.choice(ni_b, size=k, replace=False))
        if keep_m:
            keep_m = np.concatenate(keep_m); keep_n = np.concatenate(keep_n)
            mu, mi = mu[keep_m], mi[keep_m]
            nu, ni = nu[keep_n], ni[keep_n]
            print(f"[balance_by_degree] after rebalance: members={len(mu)} non-members={len(nu)}")

    ue, ie = get_inference_embeddings(model)
    # Test-time defenses applied to embeddings before MIA feature extraction.
    if args.score_temperature != 1.0:
        ue = ue / args.score_temperature
        ie = ie / args.score_temperature
    if args.predict_noise_std > 0:
        ue = ue + torch.randn_like(ue) * args.predict_noise_std
        ie = ie + torch.randn_like(ie) * args.predict_noise_std
    f_m = build_features(ue, ie, mu, mi, data_handler.trn_mat)
    f_n = build_features(ue, ie, nu, ni, data_handler.trn_mat)
    feats = np.concatenate([f_m, f_n], axis=0)
    labels = np.concatenate(
        [np.ones(len(f_m), np.int64), np.zeros(len(f_n), np.int64)], axis=0)

    metrics = train_attacker(feats, labels, args)
    print("\n==== Attack Result ====")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k:<10}: {v:.4f}")
        else:
            print(f"{k:<10}: {v}")
    print(f"\nRisk: {describe_risk(metrics['auc'])}")

    if args.out_json:
        record = {
            'kind': 'mia',
            'model': configs['model']['name'],
            'dataset': configs['data']['name'],
            'seed': configs['train']['seed'],
            'tag': configs.get('tag') or 'default',
            'attacker': args.attacker,
            'checkpoint': ckpt,
            **metrics,
        }
        # try to merge utility metrics produced by trainer
        tag = configs.get('tag') or 'default'
        tag_suffix = f"-{tag}" if tag and tag != 'default' else ''
        utility_path = os.path.join(
            'results', 'utility',
            f"utility-{configs['model']['name']}-{configs['data']['name']}-{configs['train']['seed']}{tag_suffix}.json")
        if os.path.exists(utility_path):
            with open(utility_path, 'r', encoding='utf-8') as f:
                util = json.load(f)
            for kk, vv in util.items():
                if kk not in record and kk not in ('kind', 'model', 'dataset', 'seed', 'tag'):
                    record[kk] = vv
        os.makedirs(os.path.dirname(args.out_json) or '.', exist_ok=True)
        with open(args.out_json, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
        print(f"\nAppended metrics to {args.out_json}")


if __name__ == '__main__':
    main()
