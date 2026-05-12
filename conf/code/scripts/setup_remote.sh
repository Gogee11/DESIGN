#!/usr/bin/env bash
# Set up remote env on SCNet DCU host. Run from the project root on remote.
set -eo pipefail

# 1. Activate DTK toolchain so torch.cuda works (env.sh refs unset $LD_LIBRARY_PATH so disable -u)
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}
export PATH=${PATH:-}
if [ -f /opt/dtk-25.04.2/env.sh ]; then
    source /opt/dtk-25.04.2/env.sh
elif [ -f /opt/dtk/env.sh ]; then
    source /opt/dtk/env.sh
fi

echo "[setup] python: $(which python3)"
echo "[setup] torch:  $(python3 -c 'import torch; print(torch.__version__)')"
python3 -c 'import torch; print("[setup] cuda available:", torch.cuda.is_available(), "ndev:", torch.cuda.device_count())'

# 2. Install missing pip packages (locally to avoid touching system).
pip install --user -q scikit-learn sentence-transformers datasets || true
python3 -c 'import sklearn; print("[setup] sklearn:", sklearn.__version__)'
python3 -c 'import sentence_transformers as st; print("[setup] sentence-transformers:", st.__version__)'

# 3. HF cache and offline model dir
export HF_HOME=${HF_HOME:-/root/private_data/hf_home}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-/root/private_data/hf_cache}
export SENTENCE_TRANSFORMERS_HOME=${SENTENCE_TRANSFORMERS_HOME:-/root/private_data/hf_cache}

# 4. Local BGE encoder (already cached)
if [ ! -d /root/private_data/models/bge-small-en-v1.5 ]; then
    echo "[setup] WARN: BGE encoder not found at /root/private_data/models/bge-small-en-v1.5"
fi

mkdir -p checkpoint log results/utility data
echo "[setup] done"
