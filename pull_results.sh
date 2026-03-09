# #!/bin/bash
# cd "$(dirname "$0")"

# rsync -avz --update \
#   --include='*/' \
#   --include='*.csv' \
#   --include='*.json' \
#   --include='*.jsonl' \
#   --include='*.log' \
#   --include='*.txt' \
#   --include='*.png' \
#   --include='*.jpg' \
#   --exclude='*' \
#   andes:/dartfs/rc/lab/F/FranklandS/tom/agents/ \
#   .


#!/bin/bash
cd "$(dirname "$0")"

mkdir -p pulled_folder

rsync -avz --update \
  --exclude='__pycache__/' \
  --exclude='.cache/' \
  --exclude='.ipynb_checkpoints/' \
  --exclude='.venv/' \
  --exclude='venv/' \
  --exclude='*.pyc' \
  --exclude='core.*' \
  andes:/dartfs/rc/lab/F/FranklandS/tom/agents/cache \
  pulled_folder/