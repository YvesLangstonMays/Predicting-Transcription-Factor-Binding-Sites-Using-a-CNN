#!/usr/bin/env bash
# Master script to run the entire TF binding prediction workflow
# Uses src/ modules to ensure model artifacts align with evaluation expectations

set +e

# Determine script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

print_header() { echo -e "\n${YELLOW}========== $1 ==========${NC}\n"; }
print_step()   { echo -e "${BLUE}---> $1${NC}"; }
print_success(){ echo -e "${GREEN}✓ $1${NC}"; }
print_warning(){ echo -e "${RED}! $1${NC}"; }

# Create expected directories
mkdir -p "$PROJECT_ROOT/models/{CTCF,GATA1,CEBPA,TP53}" \
         "$PROJECT_ROOT/results/{CTCF,GATA1,CEBPA,TP53}" \
         "$PROJECT_ROOT/data/raw/{genome,jaspar,encode/{CTCF,GATA1,CEBPA,TP53}}" \
         "$PROJECT_ROOT/data/processed/{CTCF,GATA1,CEBPA,TP53}" \
         "$PROJECT_ROOT/notebooks"

TFS=("CTCF" "GATA1" "CEBPA" "TP53")

print_header "VERIFYING PROJECT STRUCTURE"
if [ -f "$SCRIPT_DIR/verify_setup.py" ]; then
  python "$SCRIPT_DIR/verify_setup.py"
  [ $? -eq 0 ] && print_success "Project structure verified" || print_warning "Verification warnings"
else
  print_warning "verify_setup.py not found, skipping"
fi

print_step "Checking gdown..."
if ! command -v gdown &> /dev/null; then
  pip install gdown &>/dev/null && print_success "gdown installed" || print_warning "Could not install gdown"
else
  print_success "gdown already installed"
fi

print_header "DOWNLOADING DATA"
if [ -d "$PROJECT_ROOT/data/processed/CEBPA" ] \
  && [ "$(ls -A "$PROJECT_ROOT/data/processed/CEBPA"/*.npy 2>/dev/null)" ]; then
  print_success "Processed data already exists, skipping all downloads and preprocessing"
else
  if [ ! -f "$PROJECT_ROOT/data/raw/genome/hg38.fa" ] \
    || [ ! -d "$PROJECT_ROOT/data/raw/jaspar" ] \
    || [ ! -d "$PROJECT_ROOT/data/raw/encode" ]; then
    if [ -f "$SCRIPT_DIR/download_data.sh" ]; then
      bash "$SCRIPT_DIR/download_data.sh" \
        && print_success "Data downloaded" \
        || print_warning "download_data.sh missing or failed"
    else
      print_warning "download_data.sh not found, cannot fetch data"
    fi
  else
    print_success "Data present, skipping download"
  fi
fi

print_header "PROCESSING DATA"
for TF in "${TFS[@]}"; do
  print_step "Processing data for $TF"
  shopt -s nullglob
  beds=( "$PROJECT_ROOT/data/raw/encode/$TF"/ENCFF*.bed )
  shopt -u nullglob

  if [ -d "$PROJECT_ROOT/data/processed/$TF" ] && [ "$(ls -A "$PROJECT_ROOT/data/processed/$TF"/*.npy 2>/dev/null)" ]; then
    print_success "Processed data for $TF already exists, skipping"
    continue
  fi

  if [ ${#beds[@]} -eq 0 ]; then
    print_warning "No BED file for $TF, skipping"
    continue
  fi

  python "$PROJECT_ROOT/src/data.py" \
    --tf "$TF" \
    --jaspar-dir "$PROJECT_ROOT/data/raw/jaspar" \
    --chip-seq-file "${beds[0]}" \
    --genome "$PROJECT_ROOT/data/raw/genome/hg38.fa" \
    --output-dir "$PROJECT_ROOT/data/processed/$TF"

  [ $? -eq 0 ] && print_success "Data processed for $TF" \
                 || print_warning "Data processing failed for $TF"
done

# Prompt user whether to train models
print_header "TRAINING MODELS"
read -p "Do you want to train models? (y/N): " TRAIN_CONFIRM
if [[ "$TRAIN_CONFIRM" =~ ^[Yy]$ ]]; then
  for TF in "${TFS[@]}"; do
    print_step "Training model for $TF"
    python "$PROJECT_ROOT/src/train.py" \
      --tf "$TF" \
      --data-dir "$PROJECT_ROOT/data/processed/$TF" \
      --output-dir "$PROJECT_ROOT/models/$TF" \
      --model-type cnn \
      --batch-size 32 \
      --epochs 100 \
      --patience 10

    [ $? -eq 0 ] && print_success "Model trained for $TF" \
                   || print_warning "Training failed for $TF"
  done
else
  print_warning "Skipping training step as per user request"
fi

print_header "EVALUATING MODELS"
for TF in "${TFS[@]}"; do
  print_step "Evaluating model for $TF"

  model_file=$(ls "$PROJECT_ROOT/models/$TF"/*.h5 2>/dev/null | head -n1)
  if [ -z "$model_file" ]; then
    print_warning "No model found for $TF, skipping evaluation"
    continue
  fi

  python "$PROJECT_ROOT/src/evaluate.py" \
    --model-path "$model_file" \
    --data-dir "$PROJECT_ROOT/data/processed/$TF" \
    --output-dir "$PROJECT_ROOT/results/$TF"

  [ $? -eq 0 ] && print_success "Model evaluated for $TF" \
                 || print_warning "Evaluation failed for $TF"
done

print_header "GENERATING SUMMARY"
if [ -f "$PROJECT_ROOT/notebooks/results.ipynb" ] && command -v jupyter &> /dev/null; then
  jupyter nbconvert --to html --execute "$PROJECT_ROOT/notebooks/results.ipynb" \
    --output "$PROJECT_ROOT/notebooks/summary_report.html" \
    && print_success "Summary report generated" \
    || print_warning "Summary generation failed"
else
  print_warning "Cannot generate summary (notebook or Jupyter missing)"
fi

print_header "WORKFLOW COMPLETE"
echo "Check models/ and results/ for outputs."
