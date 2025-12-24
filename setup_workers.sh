#!/bin/bash

# Configuration
TPU_NAME="seqcond-tpu"
ZONE="us-central2-b"
REMOTE_DIR="/home/maixent/Documents/seqcond"

echo "--- Installation des dépendances sur TOUS les workers ---"

# 1. On s'assure que pip est à jour
gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --worker=all \
    --command "pip install --upgrade pip"

# 2. On installe les requirements (assurez-vous d'avoir un requirements.txt)
# Si vous n'avez pas de requirements.txt, listez les libs ici : "pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html && pip install flax optax jax-smi wandb tensorflow datasets"
gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --worker=all \
    --command "cd $REMOTE_DIR && pip install -r requirements.txt"
