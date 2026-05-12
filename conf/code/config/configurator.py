import argparse
import os
import pickle

import yaml


def parse_configure(model=None, dataset=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='lightgcn_agr', help='Model name')
    parser.add_argument('--dataset', type=str, default=None, help='Dataset name (overrides yml)')
    parser.add_argument('--device', type=str, default='cuda', help='cpu or cuda')
    parser.add_argument('--seed', type=int, default=None, help='Random seed (overrides yml)')
    parser.add_argument('--cuda', type=str, default='0', help='CUDA_VISIBLE_DEVICES')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Override dataset directory. Default: ./data/<dataset>/')

    # Hyperparameter overrides for sweeps (each maps to one loss term)
    parser.add_argument('--embedding_size', type=int, default=None,
                        help='Override model embedding_size (test-time capacity reduction).')
    parser.add_argument('--prf_weight', type=float, default=None)
    parser.add_argument('--str_weight', type=float, default=None)
    parser.add_argument('--recon_weight', type=float, default=None)
    parser.add_argument('--beta', type=float, default=None)
    parser.add_argument('--sigma', type=float, default=None)
    parser.add_argument('--reg_weight', type=float, default=None)
    parser.add_argument('--aug_top_k', type=int, default=None)
    parser.add_argument('--keep_rate', type=float, default=None)
    parser.add_argument('--mask_ratio', type=float, default=None)
    parser.add_argument('--kd_temperature', type=float, default=None)
    # Plan B (lightgcn_agr_priv) overrides
    parser.add_argument('--priv_noise_std', type=float, default=None)
    parser.add_argument('--normalize_at_predict', type=int, default=None,
                        help='1=normalize embeddings at predict, 0=do not')
    # Plan C (lightgcn_agr_conf) overrides
    parser.add_argument('--conf_weight', type=float, default=None)
    parser.add_argument('--conf_target', type=float, default=None)
    parser.add_argument('--conf_neg_target', type=float, default=None)
    parser.add_argument('--conf_target_llm_alpha', type=float, default=None)
    # Plan D (lightgcn_agr_adv) overrides
    parser.add_argument('--adv_weight', type=float, default=None)
    parser.add_argument('--adv_lambda', type=float, default=None)

    parser.add_argument('--epoch', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--patience', type=int, default=None)
    parser.add_argument('--test_step', type=int, default=None)

    # DP-SGD overrides for main_dp.py
    parser.add_argument('--dp_noise_multiplier', '--noise_multiplier', dest='dp_noise_multiplier',
                        type=float, default=None)
    parser.add_argument('--dp_clip_norm', type=float, default=None)
    parser.add_argument('--dp_delta', type=float, default=None)
    parser.add_argument('--dp_target_epsilon', type=float, default=None)

    # Tag for grouping related sweep runs
    parser.add_argument('--tag', type=str, default=None,
                        help='Free-form tag appended to log/checkpoint suffix')

    args, _ = parser.parse_known_args()

    if args.device == 'cuda':
        os.environ.setdefault('CUDA_VISIBLE_DEVICES', args.cuda)

    model_name = (model or args.model or 'default').lower()
    pre_dir = os.getcwd()
    config_path = os.path.join(pre_dir, 'config', 'models_config', f'{model_name}.yml')
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f'Config file not found: {config_path}. Please create the YAML for model {model_name}.')

    with open(config_path, encoding='utf-8') as f:
        configs = yaml.safe_load(f)

    configs['model']['name'] = configs['model']['name'].lower()
    configs.setdefault('tune', {'enable': False})
    configs.setdefault('privacy', {})
    configs['device'] = args.device
    configs['tag'] = args.tag

    if dataset:
        configs['data']['name'] = dataset
    elif args.dataset:
        configs['data']['name'] = args.dataset

    if args.data_dir:
        configs['data']['dir'] = args.data_dir

    if args.seed is not None:
        configs['train']['seed'] = args.seed

    # train overrides
    for key in ('epoch', 'batch_size', 'patience', 'test_step'):
        v = getattr(args, key)
        if v is not None:
            configs['train'][key] = v

    # model overrides (these flatten into top-level configs['model'], the dataset-specific
    # override block in the yml is *not* updated; the model's _init_dataset_config() looks
    # at the flat key as a fallback so this works).
    for key in ('prf_weight', 'str_weight', 'recon_weight', 'beta', 'sigma',
                'reg_weight', 'aug_top_k', 'keep_rate', 'mask_ratio', 'kd_temperature',
                'priv_noise_std', 'normalize_at_predict', 'embedding_size',
                'conf_weight', 'conf_target', 'conf_neg_target', 'conf_target_llm_alpha',
                'adv_weight', 'adv_lambda'):
        v = getattr(args, key)
        if v is not None:
            if key == 'normalize_at_predict':
                v = bool(v)
            configs['model'][key] = v
            ds_block = configs['model'].get(configs['data']['name'])
            if isinstance(ds_block, dict) and key in ds_block:
                ds_block[key] = v

    # DP overrides
    if args.dp_noise_multiplier is not None:
        configs['privacy']['noise_multiplier'] = args.dp_noise_multiplier
    if args.dp_clip_norm is not None:
        configs['privacy']['clip_norm'] = args.dp_clip_norm
    if args.dp_delta is not None:
        configs['privacy']['delta'] = args.dp_delta
    if args.dp_target_epsilon is not None:
        configs['privacy']['target_epsilon'] = args.dp_target_epsilon

    # Resolve dataset directory and load LLM precomputed embeddings if available.
    ds_name = configs['data']['name']
    data_dir = configs['data'].get('dir') or os.path.join(pre_dir, 'data', ds_name)
    configs['data']['dir'] = data_dir

    import numpy as np
    for cfg_key, base in [('user_embedding', 'usr_emb_np'),
                          ('item_embedding', 'itm_emb_np')]:
        npy_path = os.path.join(data_dir, base + '.npy')
        pkl_path = os.path.join(data_dir, base + '.pkl')
        if os.path.exists(npy_path):
            configs[cfg_key] = np.load(npy_path)
        elif os.path.exists(pkl_path):
            with open(pkl_path, 'rb') as f:
                configs[cfg_key] = pickle.load(f)

    return configs


configs = parse_configure()
