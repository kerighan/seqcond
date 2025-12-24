#!/bin/bash

# --- CONFIGURATION ---
# Remplacez par le nom de votre TPU et sa zone
TPU_NAME="seqcond-tpu" 
ZONE="us-central2-b"
# Le dossier où se trouve le projet SUR le TPU (doit être identique sur tous les workers)
REMOTE_DIR="/home/maixent/Documents/seqcond"
# ---------------------

CMD=$1

if [ -z "$CMD" ]; then
    echo "Usage: ./run_tpu.sh \"<python command>\""
    echo "Exemple: ./run_tpu.sh \"python3 train_jax.py --size small --jax-distributed\""
    exit 1
fi

echo "--- 1. Mise à jour du code (Git Pull) sur tous les workers ---"
gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --worker=all --command "cd $REMOTE_DIR && git pull"

echo "--- 2. Nettoyage des processus Python existants ---"
gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --worker=all --command "pkill -9 python3 || true"

echo "--- 3. Lancement de l'entraînement ---"
gcloud compute tpus tpu-vm ssh $TPU_NAME \
    --zone=$ZONE \
    --worker=all \
    --command "cd $REMOTE_DIR && source ~/.bashrc && $CMD"

