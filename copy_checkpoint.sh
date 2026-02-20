#!/bin/bash

# Usage: ./copy_checkpoint.sh <checkpoint_name> [TPU_NAME] [ZONE]
#        ./copy_checkpoint.sh --local <local_path> [TPU_NAME] [ZONE]
# Example: ./copy_checkpoint.sh --local checkpoints/seqcond-l24-d1024-th16-sh16-m2-r2-o0-a0_step280000.pkl
# Example: ./copy_checkpoint.sh --local /tmp/my_checkpoint.pkl node-v4-64 us-central2-b

# Set correct GCloud project
gcloud config set project seqcond --quiet

# Parse --local flag
FROM_LOCAL=false
if [ "$1" = "--local" ]; then
    FROM_LOCAL=true
    shift
fi

CHECKPOINT=$1
TPU_NAME="${2:-node-v4-64}"
ZONE="${3:-us-central2-b}"
REMOTE_DIR="/home/maixent/seqcond/checkpoints"
SOURCE_WORKER=1

if [ -z "$CHECKPOINT" ]; then
    echo "Usage: ./copy_checkpoint.sh <checkpoint_name> [TPU_NAME] [ZONE]"
    echo "       ./copy_checkpoint.sh --local <local_path> [TPU_NAME] [ZONE]"
    echo ""
    echo "Available checkpoints on worker $SOURCE_WORKER:"
    gcloud compute tpus tpu-vm ssh "$TPU_NAME" --zone="$ZONE" --worker=$SOURCE_WORKER \
        --command "ls -lh $REMOTE_DIR/*.pkl 2>/dev/null | awk '{print \$NF}' | xargs -n1 basename"
    exit 1
fi

if [ "$FROM_LOCAL" = true ]; then
    # --local mode: CHECKPOINT is a local file path
    LOCAL_TMP="$CHECKPOINT"
    CHECKPOINT=$(basename "$LOCAL_TMP")
    if [ ! -f "$LOCAL_TMP" ]; then
        echo "ERROR: Local file not found: $LOCAL_TMP"
        exit 1
    fi
    echo "=== Uploading local checkpoint to ALL TPU workers ==="
    echo "TPU Name: $TPU_NAME"
    echo "Zone: $ZONE"
    echo "Local file: $LOCAL_TMP ($(ls -lh $LOCAL_TMP | awk '{print $5}'))"
    echo "Remote name: $CHECKPOINT"
    echo ""
else
    LOCAL_TMP="/tmp/$CHECKPOINT"
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
fi

# Step 2: Upload to all workers in parallel
ALL_WORKERS=(0 1 2 3 4 5 6 7)
if [ "$FROM_LOCAL" = false ]; then
    # Normal mode: skip source worker (already has it)
    ALL_WORKERS=(0 2 3 4 5 6 7)
fi

# Ensure checkpoints dir exists on all workers at once
gcloud compute tpus tpu-vm ssh "$TPU_NAME" --zone="$ZONE" --worker=all \
    --command "mkdir -p $REMOTE_DIR" 2>/dev/null

# Upload function for a single worker (runs in background)
upload_worker() {
    local w=$1
    # Check if already exists
    EXISTS=$(gcloud compute tpus tpu-vm ssh "$TPU_NAME" --zone="$ZONE" --worker=$w \
        --command "[ -f $REMOTE_DIR/$CHECKPOINT ] && echo 'yes' || echo 'no'" 2>/dev/null | tail -1)
    if [ "$EXISTS" = "zizi" ]; then
        echo "Worker $w: Already has checkpoint, skipped"
        return 0
    fi
    echo "Worker $w: Uploading..."
    gcloud compute tpus tpu-vm scp "$LOCAL_TMP" "$TPU_NAME:$REMOTE_DIR/$CHECKPOINT" \
        --zone="$ZONE" --worker=$w 2>/dev/null
    if [ $? -eq 0 ]; then
        echo "Worker $w: Done ✓"
    else
        echo "Worker $w: FAILED ✗"
        return 1
    fi
}

MAX_PARALLEL=3
echo ""
echo "--- Uploading to ${#ALL_WORKERS[@]} workers (max $MAX_PARALLEL parallel) ---"
PIDS=()
FAILED=0
for i in "${ALL_WORKERS[@]}"; do
    upload_worker "$i" &
    PIDS+=($!)
    # Throttle: if we hit the limit, wait for one to finish before launching more
    if [ ${#PIDS[@]} -ge $MAX_PARALLEL ]; then
        wait -n -p DONE_PID "${PIDS[@]}" || ((FAILED++))
        PIDS=("${PIDS[@]/$DONE_PID/}")
    fi
done

# Wait for remaining uploads
for pid in "${PIDS[@]}"; do
    [ -z "$pid" ] && continue
    wait "$pid" || ((FAILED++))
done

echo ""
if [ $FAILED -gt 0 ]; then
    echo "⚠️  $FAILED worker(s) failed"
fi
if [ "$FROM_LOCAL" = false ]; then
    echo "(Local temp file kept at $LOCAL_TMP - delete manually if needed)"
fi

echo ""
echo "--- Verifying checkpoint on all workers ---"
gcloud compute tpus tpu-vm ssh "$TPU_NAME" --zone="$ZONE" --worker=all \
    --command "ls -lh $REMOTE_DIR/$CHECKPOINT 2>/dev/null || echo 'NOT FOUND'"

echo ""
echo "✅ Checkpoint copy complete!"
