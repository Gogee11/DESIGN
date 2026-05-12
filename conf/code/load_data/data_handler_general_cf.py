import os
import pickle

import numpy as np
import scipy.sparse as sp
from scipy.sparse import coo_matrix, csr_matrix
import torch as t
import torch.utils.data as data

from config.configurator import configs
from load_data.datasets_general_cf import (
    AllRankTstData,
    PairwiseTrnData,
    PairwiseWEpochFlagTrnData,
)


class DataHandlerGeneralCF:
    def __init__(self):
        cwd = os.getcwd()
        ds = configs['data']['name']
        data_dir = configs['data'].get('dir') or os.path.join(cwd, 'data', ds)
        self.data_dir = data_dir
        # Prefer .npz (cross-numpy-version stable); fall back to legacy .pkl.
        def _resolve(stem):
            for ext in ('.npz', '.pkl'):
                cand = os.path.join(data_dir, stem + ext)
                if os.path.exists(cand):
                    return cand
            return os.path.join(data_dir, stem + '.pkl')
        self.trn_file = _resolve('trn_mat')
        self.val_file = _resolve('val_mat')
        self.tst_file = _resolve('tst_mat')

    def _load_one_mat(self, file):
        if file.endswith('.npz'):
            from scipy.sparse import load_npz
            mat = load_npz(file)
        else:
            with open(file, 'rb') as fs:
                mat = pickle.load(fs)
        mat = (mat != 0).astype(np.float32)
        if not isinstance(mat, coo_matrix):
            mat = coo_matrix(mat)
        return mat

    def _normalize_adj(self, mat):
        degree = np.array(mat.sum(axis=-1))
        d_inv_sqrt = np.power(degree, -0.5).reshape(-1)
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
        d_inv_sqrt_mat = sp.diags(d_inv_sqrt)
        return mat.dot(d_inv_sqrt_mat).transpose().dot(d_inv_sqrt_mat).tocoo()

    def _make_torch_adj(self, mat, self_loop=False):
        un = configs['data']['user_num']
        im = configs['data']['item_num']
        if not self_loop:
            a = csr_matrix((un, un))
            b = csr_matrix((im, im))
        else:
            data_arr = np.ones(un)
            idx = np.arange(un)
            a = csr_matrix((data_arr, (idx, idx)), shape=(un, un))
            data_arr = np.ones(im)
            idx = np.arange(im)
            b = csr_matrix((data_arr, (idx, idx)), shape=(im, im))

        big = sp.vstack([sp.hstack([a, mat]), sp.hstack([mat.transpose(), b])])
        big = (big != 0) * 1.0
        big = self._normalize_adj(big)

        idxs = t.from_numpy(np.vstack([big.row, big.col]).astype(np.int64))
        vals = t.from_numpy(big.data.astype(np.float32))
        shape = t.Size(big.shape)
        return t.sparse_coo_tensor(idxs, vals, shape).to(configs['device'])

    def _build_aug_bipartite(self, trn_mat, top_k):
        """Augment training bipartite graph with top-K LLM-similarity edges per user.

        For each user, we look at the cosine similarity between the user's pre-computed
        LLM review embedding and all item LLM text embeddings, mask out edges already
        in the training set, and add the top_k most-similar non-train edges. The result
        is a 'soft' graph that injects LLM semantic knowledge into the GNN structure.
        """
        user_emb = configs.get('user_embedding')
        item_emb = configs.get('item_embedding')
        if user_emb is None or item_emb is None:
            return trn_mat.copy()

        un, im = trn_mat.shape
        if user_emb.shape[0] != un or item_emb.shape[0] != im:
            return trn_mat.copy()
        top_k = min(int(top_k), im - 1)
        if top_k <= 0:
            return trn_mat.copy()

        u = user_emb / (np.linalg.norm(user_emb, axis=1, keepdims=True) + 1e-8)
        v = item_emb / (np.linalg.norm(item_emb, axis=1, keepdims=True) + 1e-8)

        sim = (u @ v.T).astype(np.float32)
        train_coo = trn_mat.tocoo()
        sim[train_coo.row, train_coo.col] = -np.inf

        topk_idx = np.argpartition(-sim, top_k, axis=1)[:, :top_k]
        rows = np.repeat(np.arange(un), top_k)
        cols = topk_idx.flatten()
        vals = np.ones(len(rows), dtype=np.float32)
        aug_extra = coo_matrix((vals, (rows, cols)), shape=(un, im))

        union = (trn_mat.astype(bool) + aug_extra.astype(bool)).astype(np.float32).tocoo()
        return union

    def load_data(self):
        trn_mat = self._load_one_mat(self.trn_file)
        val_mat = self._load_one_mat(self.val_file)
        tst_mat = self._load_one_mat(self.tst_file)

        self.trn_mat = trn_mat
        configs['data']['user_num'], configs['data']['item_num'] = trn_mat.shape
        self.torch_adj = self._make_torch_adj(trn_mat)

        if configs['model']['name'] == 'gccf':
            self.torch_adj = self._make_torch_adj(trn_mat, self_loop=True)

        # Build LLM-augmented adjacency for *_agr models.
        if 'agr' in configs['model']['name'] and \
                'user_embedding' in configs and 'item_embedding' in configs:
            top_k = int(configs['model'].get('aug_top_k', 10))
            aug_bipartite = self._build_aug_bipartite(trn_mat, top_k=top_k)
            self.aug_torch_adj = self._make_torch_adj(aug_bipartite)

        if configs['train']['loss'] == 'pairwise':
            trn_data = PairwiseTrnData(trn_mat)
        elif configs['train']['loss'] == 'pairwise_with_epoch_flag':
            trn_data = PairwiseWEpochFlagTrnData(trn_mat)

        val_data = AllRankTstData(val_mat, trn_mat)
        tst_data = AllRankTstData(tst_mat, trn_mat)
        self.test_dataloader = data.DataLoader(
            tst_data, batch_size=configs['test']['batch_size'], shuffle=False, num_workers=0)
        self.valid_dataloader = data.DataLoader(
            val_data, batch_size=configs['test']['batch_size'], shuffle=False, num_workers=0)
        self.train_dataloader = data.DataLoader(
            trn_data, batch_size=configs['train']['batch_size'], shuffle=True, num_workers=0)
