#!/bin/bash

# Usage: ./run_tpu.sh "python command" [TPU_NAME] [ZONE]
# Example: ./run_tpu.sh "python train_jax.py --size small" seqcond-tpu us-central2-b

# Configuration
CMD=$1
TPU_NAME="${2:-seqcond-tpu}"
ZONE="${3:-us-central2-b}"
REMOTE_DIR="$HOME/seqcond"

if [ -z "$CMD" ]; then
    echo "Usage: ./run_tpu.sh \"<python command>\" [TPU_NAME] [ZONE]"
    echo "Example: ./run_tpu.sh \"python train_jax.py --size small --fsdp\" seqcond-tpu us-central2-b"
    exit 1
fi

echo "=== Running on TPU ==="
echo "TPU Name: $TPU_NAME"
echo "Zone: $ZONE"
echo "Command: $CMD"
echo ""

echo "--- 1. Update code (git pull) on all workers ---"
gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --worker=all \
    --command "cd $REMOTE_DIR && git pull"

echo ""
echo "--- 2. Kill existing Python processes ---"
gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --worker=all \
    --command "pkill -9 python3 || true"

echo ""
echo "--- 3. Launch training ---"
gcloud compute tpus tpu-vm ssh $TPU_NAME \
    --zone=$ZONE \
    --worker=all \
    --command "cd $REMOTE_DIR && source ~/.bashrc && $CMD"
