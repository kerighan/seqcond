#!/bin/bash

# Usage: ./run_tpu.sh "python command" [TPU_NAME] [ZONE]
# Example: ./run_tpu.sh "python train_jax.py --size small" seqcond-tpu us-central2-b

# Set correct GCloud project
gcloud config set project seqcond --quiet

# Configuration
CMD=$1
TPU_NAME="${2:-seqcond-tpu}"
ZONE="${3:-us-central2-b}"
REMOTE_DIR="$HOME/seqcond"
SESSION_NAME="train"

# Get environment variables from local machine
WANDB_KEY="${WANDB_API_KEY}"
HF_KEY="${HF_TOKEN}"

if [ -z "$CMD" ]; then
    echo "Usage: ./run_tpu.sh \"<python command>\" [TPU_NAME] [ZONE]"
    echo "Example: ./run_tpu.sh \"python train_jax.py --size small --fsdp\" seqcond-tpu us-central2-b"
    exit 1
fi

echo "=== Running on TPU ==="
echo "TPU Name: $TPU_NAME"
echo "Zone: $ZONE"
echo "Command: $CMD"
echo "Session: $SESSION_NAME (tmux)"
echo ""


echo "--- 1. Update code (git pull) on all workers ---"
gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --worker=all \
    --command "cd $REMOTE_DIR && git pull"
echo ""
echo "--- 1b. Configure wandb credentials on all workers ---"
gcloud compute tpus tpu-vm ssh "$TPU_NAME" --zone="$ZONE" --worker=all \
  --command "printf 'machine api.wandb.ai\n  login user\n  password %s\n' '$WANDB_KEY' > ~/.netrc && chmod 600 ~/.netrc" && mkdir -p ~/.cache/huggingface && echo '{\"token\":\"'$HF_KEY'\"}' > ~/.cache/huggingface/token && chmod 600 ~/.cache/huggingface/token

echo ""
echo "--- 2. Kill existing tmux sessions and Python processes ---"
gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --worker=all \
    --command "tmux kill-session -t $SESSION_NAME 2>/dev/null || true; pkill -9 python3 || true"

# echo ""
# echo "--- 3. Launch training in tmux session (with environment variables) ---"
# gcloud compute tpus tpu-vm ssh $TPU_NAME \
#     --zone=$ZONE \
#     --worker=all \
#     --command "cd $REMOTE_DIR && tmux new-session -d -s $SESSION_NAME \"export WANDB_API_KEY='$WANDB_KEY' && export HF_TOKEN='$HF_KEY' && $CMD 2>&1 | tee train.log\""
echo ""
echo "--- 3. Launch training in tmux session (with environment variables) ---"
gcloud compute tpus tpu-vm ssh "$TPU_NAME" --zone="$ZONE" --worker=all \
  --command "cd $REMOTE_DIR && tmux new-session -d -s $SESSION_NAME \
\"env WANDB_API_KEY='$WANDB_KEY' HF_TOKEN='$HF_KEY' \
     PATH=\$PATH:/home/maixent/.local/bin \
     bash -lc '$CMD 2>&1 | tee train.log'\""

echo ""
echo "✅ Training launched in tmux session '$SESSION_NAME'"
echo ""
echo "To monitor training:"
echo "  ./monitor_tpu.sh $TPU_NAME $ZONE"
echo ""
echo "To attach to the session:"
echo "  gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --worker=0 -- tmux attach -t $SESSION_NAME"
echo ""
echo "To view logs:"
echo "  gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --worker=0 -- tail -f $REMOTE_DIR/train.log"
