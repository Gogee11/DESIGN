import os
import yaml
import pickle
import argparse
import datetime


def get_available_datasets(data_root=None):
    data_root = data_root or os.path.join(os.getcwd(), "data")
    if not os.path.isdir(data_root):
        return []
    return sorted(
        name for name in os.listdir(data_root)
        if os.path.isdir(os.path.join(data_root, name))
    )


def parse_configure(model=None, dataset=None):
    available_datasets = get_available_datasets()
    default_dataset = 'digital_music' if 'digital_music' in available_datasets else (
        available_datasets[0] if available_datasets else None
    )

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='lightgcn_agr', help='Model name')
    parser.add_argument(
        '--dataset',
        type=str,
        default=default_dataset,
        choices=available_datasets if available_datasets else None,
        help='Dataset name'
    )
    parser.add_argument('--device', type=str, default='cuda', help='cpu or cuda')
    parser.add_argument('--seed', type=int, default=2025, help='Random number')
    parser.add_argument('--cuda', type=str, default='0', help='Device number')
    parser.add_argument('--diverse', type=int, default=2, help='Diverse profile number')
    parser.add_argument('--exp_tag', type=str, default=None, help='Optional experiment tag for logging/checkpoints')
    parser.add_argument('--train_batch_size', type=int, default=None, help='Optional train.batch_size override')
    parser.add_argument('--train_epoch', type=int, default=None, help='Optional train.epoch override')
    parser.add_argument('--train_patience', type=int, default=None, help='Optional train.patience override')
    parser.add_argument('--train_test_step', type=int, default=None, help='Optional train.test_step override')
    parser.add_argument('--test_batch_size', type=int, default=None, help='Optional test.batch_size override')
    parser.add_argument('--reg_weight', type=float, default=None, help='Optional model reg_weight override')
    parser.add_argument('--keep_rate', type=float, default=None, help='Optional model keep_rate override')
    parser.add_argument('--beta', type=float, default=None, help='Optional model beta override')
    parser.add_argument('--prf_weight', type=float, default=None, help='Optional model prf_weight override')
    parser.add_argument('--str_weight', type=float, default=None, help='Optional model str_weight override')
    parser.add_argument('--alpha', type=float, default=None, help='Optional model alpha override')
    parser.add_argument('--kd_temperature', type=float, default=None, help='Optional model kd_temperature override')
    parser.add_argument('--mask_ratio', type=float, default=None, help='Optional model mask_ratio override')
    parser.add_argument('--cl_weight', type=float, default=None, help='Optional model cl_weight override')
    parser.add_argument('--cen_weight', type=float, default=None, help='Optional model cen_weight override')
    parser.add_argument(
        '--dp_noise_multiplier',
        '--noise_multiplier',
        type=float,
        default=None,
        help='Optional DP noise multiplier override'
    )
    parser.add_argument('--dp_delta', type=float, default=None, help='Optional DP delta override')
    parser.add_argument('--dp_clip_norm', type=float, default=None, help='Optional DP clipping norm override')
    parser.add_argument('--dp_target_epsilon', type=float, default=None, help='Optional DP target epsilon override')
    parser.add_argument('--dp_microbatch_size', type=int, default=None, help='Optional privacy.microbatch_size override')
    parser.add_argument('--dp_num_microbatches', type=int, default=None, help='Optional privacy.num_microbatches override')
    parser.add_argument('--dp_target_scope', type=str, default=None, help='Optional privacy.target_scope override')
    parser.add_argument('--dp_disable_model_randomness', type=str, default=None, help='Optional privacy.disable_model_randomness override')
    args, _ = parser.parse_known_args()

    if args.device == 'cuda':
        os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda

    model_name = model.lower() if model else args.model.lower() if args.model else 'default'
    if dataset:
        args.dataset = dataset

    pre_dir = os.getcwd()
    config_path = f"{pre_dir}/config/models_config/{model_name}.yml"
    if not os.path.exists(config_path):
        raise Exception("Please create the yaml file for your model first.")

    with open(config_path, encoding='utf-8') as f:
        configs = yaml.safe_load(f)
    configs['model']['name'] = configs['model']['name'].lower()
    configs.setdefault('tune', {'enable': False})
    configs.setdefault('privacy', {})
    run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_tag = args.exp_tag.strip() if args.exp_tag else None
    run_id = run_timestamp if not run_tag else f"{run_timestamp}_{run_tag}"
    configs['run'] = {
        'timestamp': run_timestamp,
        'tag': run_tag,
        'id': run_id,
    }
    configs['device'] = args.device
    configs['diverse'] = args.diverse
    if args.dataset:
        configs['data']['name'] = args.dataset
    if args.seed:
        configs['train']['seed'] = args.seed
    if args.train_batch_size is not None:
        configs['train']['batch_size'] = args.train_batch_size
    if args.train_epoch is not None:
        configs['train']['epoch'] = args.train_epoch
    if args.train_patience is not None:
        configs['train']['patience'] = args.train_patience
    if args.train_test_step is not None:
        configs['train']['test_step'] = args.train_test_step
    if args.test_batch_size is not None:
        configs['test']['batch_size'] = args.test_batch_size
    if args.dp_noise_multiplier is not None:
        configs['privacy']['noise_multiplier'] = args.dp_noise_multiplier
    if args.dp_delta is not None:
        configs['privacy']['delta'] = args.dp_delta
    if args.dp_clip_norm is not None:
        configs['privacy']['clip_norm'] = args.dp_clip_norm
    if args.dp_target_epsilon is not None:
        configs['privacy']['target_epsilon'] = args.dp_target_epsilon
    if args.dp_microbatch_size is not None:
        configs['privacy']['microbatch_size'] = args.dp_microbatch_size
    if args.dp_num_microbatches is not None:
        configs['privacy']['num_microbatches'] = args.dp_num_microbatches
    if args.dp_target_scope is not None:
        configs['privacy']['target_scope'] = args.dp_target_scope
    if args.dp_disable_model_randomness is not None:
        configs['privacy']['disable_model_randomness'] = args.dp_disable_model_randomness.lower() in {'1', 'true', 'yes', 'y', 'on'}

    model_overrides = {
        'reg_weight': args.reg_weight,
        'keep_rate': args.keep_rate,
        'beta': args.beta,
        'prf_weight': args.prf_weight,
        'str_weight': args.str_weight,
        'alpha': args.alpha,
        'kd_temperature': args.kd_temperature,
        'mask_ratio': args.mask_ratio,
        'cl_weight': args.cl_weight,
        'cen_weight': args.cen_weight,
    }
    for key, value in model_overrides.items():
        if value is None:
            continue
        configs['model'][key] = value
        dataset_name = configs['data']['name']
        if dataset_name in configs['model'] and isinstance(configs['model'][dataset_name], dict):
            configs['model'][dataset_name][key] = value

    user_embedding_path = f"{pre_dir}/data/{configs['data']['name']}/usr_emb_np.pkl"
    item_embedding_path = f"{pre_dir}/data/{configs['data']['name']}/itm_emb_np.pkl"
    if not os.path.exists(user_embedding_path):
        raise FileNotFoundError(f"Missing user embedding file: {user_embedding_path}")
    if not os.path.exists(item_embedding_path):
        raise FileNotFoundError(f"Missing item embedding file: {item_embedding_path}")
    with open(user_embedding_path, 'rb') as f:
        configs['user_embedding'] = pickle.load(f)
    with open(item_embedding_path, 'rb') as f:
        configs['item_embedding'] = pickle.load(f)

    # for index in range(configs['diverse'] - 2):
    #     user_embedding_index_path = f"{pre_dir}/data/{configs['data']['name']}/diverse_profile/diverse_user_embedding_{index + 1}.pkl"
    #     item_embedding_index_path = f"{pre_dir}/data/{configs['data']['name']}/diverse_profile/diverse_item_embedding_{index + 1}.pkl"
    #     with open(user_embedding_index_path, 'rb') as f:
    #         configs[f'user_embedding_{index + 1}'] = pickle.load(f)
    #     with open(item_embedding_index_path, 'rb') as f:
    #         configs[f'item_embedding_{index + 1}'] = pickle.load(f)

    return configs

configs = parse_configure()
