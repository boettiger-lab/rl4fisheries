#!/bin/bash
# entrypoint.sh
set -e

echo "========================================================"
echo " Container for RL approaches to fishery harvest control"
echo " Package: $(python -c 'import rl4fisheries; print(rl4fisheries.__version__)')"
echo " CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "========================================================"

exec "$@"