set -uo pipefail
cd "$HOME/work/baby-model-cuda-56f0c3d"
tar -xzf "$HOME/bm-v249.tar.gz"
{
  date -u +"start %FT%TZ"
  ./.venv-minigrid-cuda/bin/python -c 'from baby_model.minigrid_torch import representation_optimizer_self_check as c; c()'
  ./.venv-minigrid-cuda/bin/python -m baby_model.minigrid_torch_sweep \
    --config configs/experiments/minigrid-torch-adda-v52.json \
    --output-dir .tmp/v249-separate-opt --seeds 5305,5306,5307,5308 --device cpu
  echo "exit=$?"
  date -u +"end %FT%TZ"
} > "$HOME/v249.log" 2>&1
tail -6 "$HOME/v249.log"
