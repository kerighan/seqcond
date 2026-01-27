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
LOCAL_TMP="/tmp/$CHECKPOINT"

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

# Step 1: Download from source worker to local machine (skip if already exists)
if [ -f "$LOCAL_TMP" ]; then
    echo "--- Local copy already exists: $LOCAL_TMP ($(ls -lh $LOCAL_TMP | awk '{print $5}')) ---"
else
    echo "--- Downloading from worker $SOURCE_WORKER to local machine ---"
    gcloud compute tpus tpu-vm scp "$TPU_NAME:$REMOTE_DIR/$CHECKPOINT" "$LOCAL_TMP" \
        --zone="$ZONE" --worker=$SOURCE_WORKER

    if [ ! -f "$LOCAL_TMP" ]; then
        echo "ERROR: Failed to download checkpoint"
        exit 1
    fi
    echo "Downloaded to $LOCAL_TMP ($(ls -lh $LOCAL_TMP | awk '{print $5}'))"
fi

# Step 2: Upload to all workers that don't have it
for i in 0 2 3 4 5 6 7; do
    echo ""
    echo "--- Checking worker $i ---"
    
    # Check if file already exists on this worker
    EXISTS=$(gcloud compute tpus tpu-vm ssh "$TPU_NAME" --zone="$ZONE" --worker=$i \
        --command "[ -f $REMOTE_DIR/$CHECKPOINT ] && echo 'yes' || echo 'no'" 2>/dev/null | tail -1)
    
    if [ "$EXISTS" = "yes" ]; then
        echo "Worker $i: Already has checkpoint, skipping"
        continue
    fi
    
    echo "Worker $i: Uploading..."
    # Ensure checkpoints directory exists
    gcloud compute tpus tpu-vm ssh "$TPU_NAME" --zone="$ZONE" --worker=$i \
        --command "mkdir -p $REMOTE_DIR"
    gcloud compute tpus tpu-vm scp "$LOCAL_TMP" "$TPU_NAME:$REMOTE_DIR/$CHECKPOINT" \
        --zone="$ZONE" --worker=$i
done

echo ""
echo "(Local temp file kept at $LOCAL_TMP - delete manually if needed)"

echo ""
echo "--- Verifying checkpoint on all workers ---"
gcloud compute tpus tpu-vm ssh "$TPU_NAME" --zone="$ZONE" --worker=all \
    --command "ls -lh $REMOTE_DIR/$CHECKPOINT 2>/dev/null || echo 'NOT FOUND'"

echo ""
echo "✅ Checkpoint copy complete!"
