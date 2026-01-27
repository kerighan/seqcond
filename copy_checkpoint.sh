#!/bin/bash

# Usage: ./copy_checkpoint.sh <checkpoint_name> [TPU_NAME] [ZONE]
# Example: ./copy_checkpoint.sh seqcond-l24-d1024-th16-sh16-m2-r2-o0-a0_step20000.pkl

# Set correct GCloud project
gcloud config set project seqcond --quiet

CHECKPOINT=$1
TPU_NAME="${2:-node-v4-64}"
ZONE="${3:-us-central2-b}"
REMOTE_DIR="/home/maixent/seqcond/checkpoints"
SOURCE_WORKER=1

if [ -z "$CHECKPOINT" ]; then
    echo "Usage: ./copy_checkpoint.sh <checkpoint_name> [TPU_NAME] [ZONE]"
    echo ""
    echo "Available checkpoints on worker $SOURCE_WORKER:"
    gcloud compute tpus tpu-vm ssh "$TPU_NAME" --zone="$ZONE" --worker=$SOURCE_WORKER \
        --command "ls -lh $REMOTE_DIR/*.pkl 2>/dev/null | awk '{print \$NF}' | xargs -n1 basename"
    exit 1
fi

echo "=== Copying checkpoint across TPU workers ==="
echo "TPU Name: $TPU_NAME"
echo "Zone: $ZONE"
echo "Checkpoint: $CHECKPOINT"
echo "Source worker: $SOURCE_WORKER"
echo ""

# Get the IP of worker 1 (source)
echo "--- Getting source worker IP ---"
SOURCE_IP=$(gcloud compute tpus tpu-vm describe "$TPU_NAME" --zone="$ZONE" \
    --format="value(networkEndpoints[$SOURCE_WORKER].ipAddress)")
echo "Source worker $SOURCE_WORKER IP: $SOURCE_IP"

# Copy from worker 1 to all other workers using scp
for i in 0 2 3 4 5 6 7; do
    echo ""
    echo "--- Copying to worker $i ---"
    gcloud compute tpus tpu-vm ssh "$TPU_NAME" --zone="$ZONE" --worker=$i \
        --command "scp -o StrictHostKeyChecking=no maixent@$SOURCE_IP:$REMOTE_DIR/$CHECKPOINT $REMOTE_DIR/$CHECKPOINT"
done

echo ""
echo "--- Verifying checkpoint on all workers ---"
gcloud compute tpus tpu-vm ssh "$TPU_NAME" --zone="$ZONE" --worker=all \
    --command "ls -lh $REMOTE_DIR/$CHECKPOINT 2>/dev/null || echo 'NOT FOUND'"

echo ""
echo "✅ Checkpoint copy complete!"
