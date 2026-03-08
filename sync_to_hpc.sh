#!/bin/bash
cd "$(dirname "$0")"

while true; do rsync -avz \
  --exclude '.git' \
  --exclude '.venv' \
  --exclude 'venv' \
  --exclude '.vscode' \
  --exclude '.conda' \
  --exclude '.conda_backup_20250713_101749' \
  --exclude '__pycache__' \
  --exclude '.pytest_cache' \
  --exclude '.ipynb_checkpoints' \
  --exclude 'results' \
  --exclude 'outputs' \
  --exclude 'wandb' \
  --exclude 'paper' \
  --exclude '*.pt' \
  --exclude '*.pth' \
  --exclude '*.log' \
  --exclude '*.csv' \
  --exclude '*.md' \
  . andes:/dartfs/rc/lab/F/FranklandS/tom/agents; sleep 2; done
