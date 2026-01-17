#!/bin/bash

# Usage: ./setup_workers.sh [TPU_NAME] [ZONE]
# Example: ./setup_workers.sh seqcond-tpu us-central2-b

# Set correct GCloud project
gcloud config set project seqcond --quiet

# Configuration
TPU_NAME="${1:-node-v4-64}"
ZONE="${2:-us-central2-b}"
REPO_URL="https://github.com/kerighan/seqcond"
REMOTE_DIR="$HOME/seqcond"

echo "=== Setup TPU Workers ==="
echo "TPU Name: $TPU_NAME"
echo "Zone: $ZONE"
echo "Remote Dir: $REMOTE_DIR"
echo ""

echo "--- 1. Upgrade pip on all workers ---"
gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --worker=all \
    --command "pip install --upgrade pip"

echo ""
echo "--- 2. Clone repository (or update if exists) ---"
gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --worker=all \
    --command "if [ -d $REMOTE_DIR ]; then cd $REMOTE_DIR && git pull; else git clone $REPO_URL $REMOTE_DIR; fi"

echo ""
echo "--- 3. Install JAX for TPU ---"
gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --worker=all \
    --command "pip install 'jax[tpu]' -f https://storage.googleapis.com/jax-releases/libtpu_releases.html"

echo ""
echo "--- 4. Install requirements.txt ---"
gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --worker=all \
    --command "cd $REMOTE_DIR && pip install -r requirements.txt"

echo ""
echo "--- 5. Install jax-smi for monitoring ---"
gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --worker=all \
    --command "pip install jax-smi"

echo ""
echo "=== Setup complete! ==="
echo "You can now run: ./run_tpu.sh \"python train_jax.py ...\""
