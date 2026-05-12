
import torch
import random
import numpy as np
from logging import getLogger

def set_seed(seed):
    # On torch 2.4.x DTK builds torch.manual_seed() *internally* calls
    # cuda.manual_seed_all() which queues a deferred call that errors out at
    # the next CUDA op with "tuple index out of range". We monkey-patch
    # manual_seed_all to a no-op before calling manual_seed to avoid this.
    try:
        major, minor = torch.__version__.split('.')[:2]
        if (int(major), int(minor)) < (2, 5):
            import torch.cuda as _tc
            _tc.manual_seed_all = lambda s: None  # no-op
    except Exception:
        pass

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        try:
            torch.cuda.manual_seed(seed)
        except Exception as e:
            print(f"[set_seed] cuda manual_seed failed (non-fatal): {e}")

class DisabledSummaryWriter:
    def __init__(*args, **kwargs):
        pass
    def __call__(self, *args, **kwargs):
        return self
    def __getattr__(self, *args, **kwargs):
        return self

def log_exceptions(func):
    def wrapper(*args, **kwargs):
        logger = getLogger('train_logger')
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.exception(e)
            raise e
    return wrapper