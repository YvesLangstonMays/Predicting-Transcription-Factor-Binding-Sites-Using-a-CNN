# TF Binding Prediction Project

## Table of Contents

1. [Introduction](#introduction)
2. [Project Structure](#project-structure)
3. [Biological Background](#biological-background)
4. [QuickStart Guide](#quick-start-guide)
5. [Data](#data)
6. [Running The Workflow](#run-full-workflow)
7. [Dependencies](#obscure-package-descriptions-w-examples-ai-generated)
   - [pyfaidx](#pyfaidx)
   - [MLflow](#mlflow)
   - [pybedtools](#pybedtools)
   - [pysam](#pysam)
   - [SHAP](#shap-shapley-additive-explanations)
   - [Hydra-core](#hydra-core)
   - [pytest](#pytest)
   - [pytest-cov](#pytest-cov)
   - [Sphinx](#sphinx)

# Project Structure

```bash
tf_binding_prediction/
├── README.md                  # Project documentation
├── requirements.txt           # Project dependencies
├── data/
│   ├── raw/                   # Original data from JASPAR
│   └── processed/             # Processed data (one-hot encoded sequences)
├── notebooks/                 # Analysis notebooks
│   ├── exploratory.ipynb      # Data exploration
│   └── results.ipynb          # Results visualization
├── src/                       # Source code
│   ├── data.py                # Data loading and preprocessing
│   ├── model.py               # CNN and baseline model implementations
│   ├── train.py               # Training functions
│   └── evaluate.py            # Evaluation metrics and functions
├── scripts/
│   ├── download_data.sh       # Script to download data
│   ├── train_model.py         # Script to train the model
│   └── evaluate_model.py      # Script to evaluate the model
├── tests/                     # Basic tests for critical functionality
│   └── test_data.py           # Test data processing functions
└── config.yaml                # Single configuration file
```

# Biological Background

## Transcription Factors and Gene Regulation

### DNA, Genes, and Proteins

- **DNA** is the molecule that contains the genetic instructions for development and functioning of all living organisms
- **Genes** are segments of DNA that contain instructions for making specific proteins
- **Proteins** are complex molecules that perform most cellular functions and make up most cellular structures

### Transcription Factors (TFs)

- **Transcription factors** are specialized proteins that control which genes are "turned on" or "turned off"
- They work by binding to specific short DNA sequences (6-20 base pairs) near genes
- These binding events help determine when and how much a gene is expressed
- Think of TFs as biological "switches" that control gene activation

### Why Predicting TF Binding Sites Matters

- Knowing where TFs bind helps us understand how cells regulate their genes
- Mutations in TF binding sites can cause diseases by disrupting normal gene regulation
- Many drugs target transcription factor pathways

### The Computational Challenge

- The human genome contains ~3 billion DNA bases
- Finding the short sequences (200bp) where specific TFs bind is like finding needles in a haystack
- Experimental methods can identify binding sites but are expensive and time consuming
- Our CNN model aims to learn the complex DNA sequence patterns that TFs recognize

### Data Representation

- DNA sequences consist of four nucleotides: A (Adenine), C (Cytosine), G (Guanine), and T (Thymine)
- We represent these as "one-hot encoded" vectors:
  - A = [1,0,0,0]
  - C = [0,1,0,0]
  - G = [0,0,1,0]
  - T = [0,0,0,1]
- A 200bp sequence becomes a 200×4 matrix, which is ideal for processing with CNNs

# Quick Start Guide

## Running the Workflow

The project includes a master script that handles the entire workflow from data download to model evaluation.

### Shrunken dataset is included in the repo due to size limitations, if you want to run the full dataset, delete the existing processed data and the script will download the full dataset + process it.

```bash
# Navigate to the project root
cd Predicting-Transcription-Factor-Binding-Sites-Using-a-CNN

# Navigate to the source code
cd project_files

# install required python packages
pip install -r requirements.txt

# navigate to scripts directory
cd scripts

# run the master workflow script
./run_workflow.sh
```

# Make the workflow script executable if needed
chmod +x scripts/run_workflow.sh

The workflow script will:

1. Create necessary directory structures
2. Download required data (reference genome, JASPAR motifs, ChIP-seq data)
3. Process data for each transcription factor
4. Train models
5. Evaluate models 
6. Generate a summary report 


# Data

CTCF (CCCTC-binding factor) is a super important transcription factor that acts like a **genomic organizer**. It binds to DNA and helps regulate the 3D structure of the genome by forming **chromatin loops**—basically bringing distant parts of the genome together or keeping them apart.

Breakdown:

- **Sequence-specific**: It binds to a specific DNA motif that's relatively easy to model computationally
- **Insulator function**: It can block the interaction between enhancers and promoters when needed, preventing the wrong genes from being turned on.
- **Master of boundaries**: CTCF marks the edges of **topologically associating domains (TADs)**—big regions of the genome that interact more with themselves than with others.
- **Involved in looping**: Often works with cohesin to create **loops** in the genome that are crucial for gene regulation.

Because it has a clear motif and tons of ChIP seq data across many cell types, it’s one of the most studied TFs and a go to for computational biology.

# Run Full Workflow

Run the entire workflow from data download to model evaluation with a single script:

```bash
scripts/run_workflow.sh
This master script will:

Verify the project structure
Download all required data
Process data for each transcription factor
Train models for each TF
Evaluate models and analyze motifs
Generate a summary report
```

# Obscure Package Descriptions w/ Examples (AI generated)

### pyfaidx

**Detailed Use Case:** When working with reference genomes or DNA sequence datasets, it is often necessary to extract specific regions around potential binding sites without loading gigabytes of sequence data into memory.

**Expanded Example:**

```python
from pyfaidx import Fasta
import numpy as np
from tqdm import tqdm

# Open and index the reference genome
genome = Fasta('hg38.fa')

# Extract sequences for a list of potential binding sites
binding_sites = [
    ("chr1", 1000000, 1000200),  # chr, start, end
    ("chr2", 2345600, 2345800),
    # Hundreds or thousands more sites
]

extracted_sequences = []
for chrom, start, end in tqdm(binding_sites, desc="Extracting sequences"):
    if chrom in genome:
        # Extract the 200bp sequence
        seq = str(genome[chrom][start:end])

        # Skip sequences containing ambiguous nucleotides
        if 'N' not in seq:
            extracted_sequences.append(seq)

print(f"Extracted {len(extracted_sequences)} valid sequences")

```

### MLflow

**Expanded Example:**

```python
import mlflow
import tensorflow as tf
import numpy as np
from sklearn.metrics import roc_auc_score

# Set the tracking URI if using a shared server
mlflow.set_tracking_uri("http://your-tracking-server:5000")
mlflow.set_experiment("TF_Binding_CNN")

# Define model architecture with hyperparameters
def create_model(conv_layers=2, filters=32, kernel_size=8, dropout_rate=0.3):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(200, 4)))  # 200bp, one-hot encoded

    # Add convolutional layers
    for i in range(conv_layers):
        model.add(tf.keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, activation='relu'))
        model.add(tf.keras.layers.MaxPooling1D(pool_size=2))
        model.add(tf.keras.layers.Dropout(dropout_rate))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dropout(dropout_rate))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    return model

# Start an MLflow run when training a model
with mlflow.start_run(run_name="CNN_2layer"):
    # Log parameters
    params = {
        "conv_layers": 2,
        "filters": 32,
        "kernel_size": 8,
        "dropout_rate": 0.3,
        "learning_rate": 0.001,
        "batch_size": 64,
        "epochs": 20,
        "transcription_factor": "CTCF"
    }

    mlflow.log_params(params)

    # Create and compile the model
    model = create_model(**{k: v for k, v in params.items() if k in ["conv_layers", "filters", "kernel_size", "dropout_rate"]})
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=params["learning_rate"]),
                 loss='binary_crossentropy',
                 metrics=['accuracy'])

    # Train the model
    history = model.fit(
        X_train, y_train,
        batch_size=params["batch_size"],
        epochs=params["epochs"],
        validation_data=(X_val, y_val),
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)]
    )

    # Log metrics from training
    for epoch, (loss, acc, val_loss, val_acc) in enumerate(zip(
            history.history['loss'],
            history.history['accuracy'],
            history.history['val_loss'],
            history.history['val_accuracy'])):
        mlflow.log_metrics({
            "loss": loss,
            "accuracy": acc,
            "val_loss": val_loss,
            "val_accuracy": val_acc
        }, step=epoch)

    # Evaluate on test set
    y_pred = model.predict(X_test)
    test_auc = roc_auc_score(y_test, y_pred)
    mlflow.log_metric("test_auc", test_auc)

    # Log the model
    mlflow.tensorflow.log_model(model, "model")

    # Log a visualization of ROC curve
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve

    fpr, tpr, _ = roc_curve(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve (AUC = {test_auc:.3f})')
    plt.savefig("roc_curve.png")

    mlflow.log_artifact("roc_curve.png")
```

### pybedtools

**Detailed Use Case:** When analyzing transcription factor binding sites, it is often necessary to relate them to genomic features like promoters, enhancers, or other regulatory elements to understand binding patterns.

**Expanded Example:**

```python
from pybedtools import BedTool
import pandas as pd
import matplotlib.pyplot as plt

# Load predicted binding sites from your model
predicted_sites = BedTool('predicted_binding_sites.bed')

# Load genomic annotations
promoters = BedTool('promoters.bed')
enhancers = BedTool('enhancers.bed')
gene_bodies = BedTool('gene_bodies.bed')
intergenic = BedTool('intergenic.bed')

# Find overlaps with different genomic features
promoter_overlaps = predicted_sites.intersect(promoters, wa=True)
enhancer_overlaps = predicted_sites.intersect(enhancers, wa=True)
gene_body_overlaps = predicted_sites.intersect(gene_bodies, wa=True)
intergenic_overlaps = predicted_sites.intersect(intergenic, wa=True)

# Calculate distribution statistics
total_sites = len(predicted_sites)
distribution = {
    "Promoters": len(promoter_overlaps) / total_sites * 100,
    "Enhancers": len(enhancer_overlaps) / total_sites * 100,
    "Gene Bodies": len(gene_body_overlaps) / total_sites * 100,
    "Intergenic": len(intergenic_overlaps) / total_sites * 100
}

# Calculate enrichment over background
genome_size = 3.2e9  # Human genome size
promoter_size = sum(p.length for p in promoters)
enhancer_size = sum(e.length for e in enhancers)
gene_body_size = sum(g.length for g in gene_bodies)
intergenic_size = sum(i.length for i in intergenic)

enrichment = {
    "Promoters": (len(promoter_overlaps) / total_sites) / (promoter_size / genome_size),
    "Enhancers": (len(enhancer_overlaps) / total_sites) / (enhancer_size / genome_size),
    "Gene Bodies": (len(gene_body_overlaps) / total_sites) / (gene_body_size / genome_size),
    "Intergenic": (len(intergenic_overlaps) / total_sites) / (intergenic_size / genome_size)
}

# Visualize the distribution and enrichment
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Distribution pie chart
ax1.pie(distribution.values(), labels=distribution.keys(), autopct='%1.1f%%')
ax1.set_title('Distribution of Predicted Binding Sites')

# Enrichment bar chart
ax2.bar(enrichment.keys(), enrichment.values())
ax2.axhline(y=1, color='r', linestyle='-')
ax2.set_ylabel('Fold Enrichment')
ax2.set_title('Enrichment Over Genomic Background')

plt.tight_layout()
plt.savefig('binding_site_analysis.png')
```

### pysam

**Detailed Use Case:** If validating predicted binding sites against ChIPseq or ATACseq data, you'll need to analyze alignment files to quantify the overlap between your predictions and experimental evidence.

**Expanded Example:**

```python
import pysam
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Function to calculate read coverage around binding sites
def calculate_coverage_profile(bamfile, binding_sites, window=1000):
    """
    Calculate average read coverage around binding sites

    Parameters:
    - bamfile: path to indexed BAM file
    - binding_sites: list of (chrom, position) tuples
    - window: size of window around binding site center

    Returns:
    - Average coverage profile array
    """
    bam = pysam.AlignmentFile(bamfile, "rb")

    # Initialize coverage array
    coverage_matrix = np.zeros((len(binding_sites), 2*window))

    for i, (chrom, pos) in enumerate(binding_sites):
        start = max(0, pos - window)
        end = pos + window

        # Get coverage for this region
        coverage = np.zeros(2*window)

        for column in bam.pileup(chrom, start, end, truncate=True):
            rel_pos = column.pos - start
            if 0 <= rel_pos < 2*window:
                coverage[rel_pos] = column.n

        coverage_matrix[i] = coverage

    # Calculate average coverage
    avg_coverage = np.mean(coverage_matrix, axis=0)

    return avg_coverage

# Load your predicted binding sites
predictions_df = pd.read_csv('predicted_binding_sites.csv')
predicted_sites = list(zip(predictions_df['chromosome'], predictions_df['position']))

# Load positive and negative control sites (e.g., experimentally validated)
positive_sites = list(zip(pd.read_csv('validated_sites.csv')['chromosome'],
                        pd.read_csv('validated_sites.csv')['position']))
negative_sites = list(zip(pd.read_csv('non_binding_sites.csv')['chromosome'],
                        pd.read_csv('non_binding_sites.csv')['position']))

# Calculate coverage profiles from ChIP-seq data
chip_seq_bam = "chip_seq_experiment.bam"
predicted_coverage = calculate_coverage_profile(chip_seq_bam, predicted_sites)
positive_coverage = calculate_coverage_profile(chip_seq_bam, positive_sites)
negative_coverage = calculate_coverage_profile(chip_seq_bam, negative_sites)

# Plot the coverage profiles
x = np.arange(-1000, 1000)
plt.figure(figsize=(10, 6))
plt.plot(x, predicted_coverage, label='Model Predictions')
plt.plot(x, positive_coverage, label='Validated Sites')
plt.plot(x, negative_coverage, label='Non-binding Sites')
plt.axvline(x=0, color='black', linestyle='--')
plt.legend()
plt.xlabel('Distance from Site Center (bp)')
plt.ylabel('Average Read Coverage')
plt.title('ChIP-seq Coverage Around Binding Sites')
plt.savefig('chip_seq_validation.png')
```

### SHAP (SHapley Additive exPlanations)

**Detailed Use Case:** After training the CNN model, understand which nucleotide positions and patterns in the 200bp sequences are most influential for predicting transcription factor binding.

**Expanded Example:**

```python
import shap
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Load your trained model and test data
model = tf.keras.models.load_model('tf_binding_model.h5')
X_test = np.load('X_test.npy')  # Shape: (n_samples, 200, 4)
sequences = np.load('test_sequences.npy')  # Original sequences

# Create a background dataset (subset of training data)
background = X_test[:100]  # Use 100 examples as background

# Create the DeepExplainer
explainer = shap.DeepExplainer(model, background)

# Calculate SHAP values for the first 500 test examples
shap_values = explainer.shap_values(X_test[:500])

# shap_values shape: (1, 500, 200, 4) -> (samples, sequence_length, nucleotides)
# Reshape to (500, 200, 4)
shap_values = shap_values[0]

# Calculate nucleotide importance
sequence_importance = np.sum(np.abs(shap_values), axis=2)  # Sum over nucleotides
avg_importance = np.mean(sequence_importance, axis=0)  # Average over samples

# Plot average importance across positions
plt.figure(figsize=(15, 5))
plt.bar(range(len(avg_importance)), avg_importance)
plt.xlabel('Position in 200bp Sequence')
plt.ylabel('Average |SHAP Value|')
plt.title('Importance of Each Position for TF Binding Prediction')
plt.savefig('position_importance.png')

# Generate sequence logo-like visualization for most important region
# Find the most important region (e.g., 20bp window)
window_size = 20
importance_sum = np.array([np.sum(avg_importance[i:i+window_size]) for i in range(len(avg_importance)-window_size)])
most_important_start = np.argmax(importance_sum)
most_important_end = most_important_start + window_size

# Get SHAP values for this region
region_shap = shap_values[:, most_important_start:most_important_end, :]

# Create sequence logo
plt.figure(figsize=(15, 5))
# For each position in the important region
for i in range(window_size):
    pos = i + most_important_start
    height = 0
    # For each nucleotide (A, C, G, T)
    for nuc in range(4):
        # Calculate mean absolute SHAP value
        mean_effect = np.mean(np.abs(region_shap[:, i, nuc]))
        plt.bar(pos, mean_effect, bottom=height, width=0.8,
                color=['green', 'blue', 'orange', 'red'][nuc])
        height += mean_effect

plt.xticks(range(most_important_start, most_important_end))
plt.xlabel('Sequence Position')
plt.ylabel('Mean |SHAP Value|')
plt.title(f'Nucleotide Importance in Region {most_important_start}-{most_important_end}')
plt.legend(['A', 'C', 'G', 'T'])
plt.savefig('nucleotide_importance.png')
```

### Hydra-core

**Detailed Use Case:** You need to experiment with different model architectures, data preprocessing approaches, and training configurations while keeping track of all parameters systematically.

**Expanded Example:**

```python
# Directory structure:
# config/
#   config.yaml     # Base configuration
#   model/
#     cnn.yaml      # CNN model configs
#     baseline.yaml # Baseline model configs
#   data/
#     preprocessing.yaml  # Data preprocessing configs
#   training/
#     default.yaml  # Default training params

# config/config.yaml
defaults:
  - model: cnn
  - data: preprocessing
  - training: default
  - _self_

experiment_name: ${model.name}_${data.preprocessing_type}
output_dir: ./outputs/${experiment_name}

# config/model/cnn.yaml
name: cnn_model
architecture:
  input_shape: [200, 4]
  conv_layers:
    - filters: 32
      kernel_size: 8
      activation: relu
      pool_size: 2
    - filters: 64
      kernel_size: 4
      activation: relu
      pool_size: 2
  dense_layers:
    - units: 64
      activation: relu
      dropout: 0.3
  output_activation: sigmoid

# config/data/preprocessing.yaml
preprocessing_type: one_hot
sequence_length: 200
negative_sampling:
  method: random_shuffle
  ratio: 1
data_augmentation:
  enabled: true
  reverse_complement: true
  random_shift: 10

# config/training/default.yaml
optimizer:
  name: adam
  learning_rate: 0.001
batch_size: 64
epochs: 100
early_stopping:
  enabled: true
  patience: 10
  monitor: val_loss
class_weights:
  enabled: true

# train.py
import hydra
from omegaconf import DictConfig, OmegaConf
import tensorflow as tf
import os
import logging
import mlflow

@hydra.main(config_path="config", config_name="config")
def train(cfg: DictConfig):
    # Set up logging
    logging.info(f"Training with config:\n{OmegaConf.to_yaml(cfg)}")

    # Set up MLflow tracking
    mlflow.set_experiment(cfg.experiment_name)
    with mlflow.start_run():
        # Log all configs
        mlflow.log_params({f"{key}.{subkey}": val
                          for key, dict_val in OmegaConf.to_container(cfg).items()
                          for subkey, val in dict_val.items()
                          if isinstance(dict_val, dict)})

        # Load data with preprocessing config
        X_train, y_train, X_val, y_val, X_test, y_test = load_data(cfg.data)

        # Build model from config
        model = build_model_from_config(cfg.model)

        # Set up optimizer
        if cfg.training.optimizer.name == "adam":
            optimizer = tf.keras.optimizers.Adam(learning_rate=cfg.training.optimizer.learning_rate)
        elif cfg.training.optimizer.name == "sgd":
            optimizer = tf.keras.optimizers.SGD(learning_rate=cfg.training.optimizer.learning_rate)

        # Compile model
        model.compile(optimizer=optimizer,
                     loss="binary_crossentropy",
                     metrics=["accuracy", tf.keras.metrics.AUC(name="auc")])

        # Set up callbacks
        callbacks = []
        if cfg.training.early_stopping.enabled:
            callbacks.append(tf.keras.callbacks.EarlyStopping(
                patience=cfg.training.early_stopping.patience,
                monitor=cfg.training.early_stopping.monitor,
                restore_best_weights=True
            ))

        # Set up class weights if enabled
        class_weights = None
        if cfg.training.class_weights.enabled:
            n_pos = sum(y_train)
            n_neg = len(y_train) - n_pos
            class_weights = {0: len(y_train)/(2*n_neg), 1: len(y_train)/(2*n_pos)}

        # Train model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            batch_size=cfg.training.batch_size,
            epochs=cfg.training.epochs,
            callbacks=callbacks,
            class_weight=class_weights
        )

        # Evaluate model
        results = model.evaluate(X_test, y_test)
        metrics = dict(zip(model.metrics_names, results))

        # Log metrics
        mlflow.log_metrics(metrics)

        # Save model
        model_path = os.path.join(cfg.output_dir, "model.h5")
        model.save(model_path)
        mlflow.log_artifact(model_path)

        logging.info(f"Training completed. Test metrics: {metrics}")

if __name__ == "__main__":
    train()
```

### pytest

**Detailed Use Case:** Ensure that critical components like the data processing pipeline and model architecture work consistently across different machines and as the code evolves.

**Expanded Example:**

```python
# src/data.py
import numpy as np

def one_hot_encode(sequence):
    """Convert DNA sequence to one-hot encoding"""
    mapping = {'A': [1,0,0,0], 'C': [0,1,0,0], 'G': [0,0,1,0], 'T': [0,0,0,1], 'N': [0,0,0,0]}
    return np.array([mapping.get(nuc.upper(), [0,0,0,0]) for nuc in sequence])

def reverse_complement(sequence):
    """Get reverse complement of DNA sequence"""
    complement = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A', 'N': 'N'}
    return ''.join(complement.get(nuc.upper(), 'N') for nuc in reversed(sequence))

def generate_negative_samples(positive_sequences, method='shuffle'):
    """Generate negative samples from positive sequences"""
    if method == 'shuffle':
        import random
        negatives = []
        for seq in positive_sequences:
            seq_list = list(seq)
            random.shuffle(seq_list)
            negatives.append(''.join(seq_list))
        return negatives
    elif method == 'dinucleotide_shuffle':
        # Preserve dinucleotide frequencies
        # Implementation here
        pass
    else:
        raise ValueError(f"Unknown method: {method}")

# tests/test_data.py
import pytest
import numpy as np
from src.data import one_hot_encode, reverse_complement, generate_negative_samples

class TestDataProcessing:

    def test_one_hot_encode_valid_sequence(self):
        """Test one-hot encoding with valid DNA sequence"""
        seq = "ACGT"
        encoded = one_hot_encode(seq)

        # Check shape
        assert encoded.shape == (4, 4)

        # Check nucleotide encodings
        assert np.array_equal(encoded[0], [1, 0, 0, 0])  # A
        assert np.array_equal(encoded[1], [0, 1, 0, 0])  # C
        assert np.array_equal(encoded[2], [0, 0, 1, 0])  # G
        assert np.array_equal(encoded[3], [0, 0, 0, 1])  # T

    def test_one_hot_encode_with_n(self):
        """Test one-hot encoding with N (unknown base)"""
        seq = "ACGTN"
        encoded = one_hot_encode(seq)

        # Check N encoding
        assert np.array_equal(encoded[4], [0, 0, 0, 0])  # N

    def test_one_hot_encode_lowercase(self):
        """Test one-hot encoding with lowercase letters"""
        seq = "acgt"
        encoded = one_hot_encode(seq)

        # Check nucleotide encodings are case-insensitive
        assert np.array_equal(encoded[0], [1, 0, 0, 0])  # a
        assert np.array_equal(encoded[1], [0, 1, 0, 0])  # c
        assert np.array_equal(encoded[2], [0, 0, 1, 0])  # g
        assert np.array_equal(encoded[3], [0, 0, 0, 1])  # t

    def test_reverse_complement(self):
        """Test reverse complement function"""
        seq = "ATGCN"
        rev_comp = reverse_complement(seq)

        assert rev_comp == "NGCAT"

        # Test palindromic sequence
        palindrome = "ACGT"
        assert reverse_complement(palindrome) == "ACGT"

    def test_generate_negative_samples_shuffle(self):
        """Test generation of negative samples via shuffling"""
        positive_seqs = ["ATGC", "CCGG"]
        negatives = generate_negative_samples(positive_seqs, method='shuffle')

        # Check we get same number of samples
        assert len(negatives) == len(positive_seqs)

        # Check same nucleotide composition but different order
        for pos, neg in zip(positive_seqs, negatives):
            assert sorted(pos) == sorted(neg)
            assert pos != neg or pos == neg * len(pos)  # Only equal if all same letter

    @pytest.mark.parametrize("method", ["shuffle", "dinucleotide_shuffle"])
    def test_generate_negative_samples_methods(self, method):
        """Test different methods of negative sample generation"""
        positive_seqs = ["ATGCATGC", "CCGGTTAA"]
        negatives = generate_negative_samples(positive_seqs, method=method)

        # Basic checks regardless of method
        assert len(negatives) == len(positive_seqs)
        assert all(len(neg) == len(pos) for neg, pos in zip(negatives, positive_seqs))

    def test_generate_negative_samples_invalid_method(self):
        """Test error handling for invalid method"""
        with pytest.raises(ValueError):
            generate_negative_samples(["ATGC"], method="invalid_method")
```

### pytest-cov

**Detailed Use Case:** Maintain code quality by ensuring that tests cover all critical paths in the codebase, especially the data processing and model components.

**Expanded Example:**

```bash
# Run tests with coverage and generate a report
pytest --cov=src --cov-report=html tests/

# This will generate an HTML report in htmlcov/ with color-coded coverage
```

**Coverage HTML Example:**
The HTML report would show:

- Overall coverage percentage
- File by file coverage with color coding (green for covered, red for uncovered)
- Line by line highlighting showing which specific code paths are tested or not

**Integration with CI/CD:**

```yaml
# .github/workflows/tests.yml
name: Tests

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.9

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt

      - name: Run tests with coverage
        run: |
          pytest --cov=src --cov-report=xml tests/

      - name: Upload coverage report
        uses: codecov/codecov-action@v1
        with:
          file: ./coverage.xml
          fail_ci_if_error: true
          verbose: true

      - name: Check coverage threshold
        run: |
          coverage report --fail-under=80
```

### Sphinx

**Detailed Use Case:** Good documentation is essential so everyone understands the codebase and can effectively collaborate. This is especially important for complex functions like model architecture and data augmentation.

**Expanded Example:**

```python
# src/model.py
def create_binding_cnn(sequence_length=200, num_filters=(32, 64),
                      kernel_sizes=(8, 4), pool_sizes=(2, 2),
                      dense_units=64, dropout_rate=0.3):
    """
    Create a 1D CNN model for transcription factor binding prediction.

    This model takes one-hot encoded DNA sequences and predicts whether a
    transcription factor binding site is present in the sequence.

    Parameters
    ----------
    sequence_length : int, optional
        Length of input DNA sequences, by default 200
    num_filters : tuple of int, optional
        Number of filters in each convolutional layer, by default (32, 64)
    kernel_sizes : tuple of int, optional
        Size of kernels in each convolutional layer, by default (8, 4)
    pool_sizes : tuple of int, optional
        Size of pooling windows in each pooling layer, by default (2, 2)
    dense_units : int, optional
        Number of units in the dense layer, by default 64
    dropout_rate : float, optional
        Dropout rate for regularization, by default 0.3

    Returns
    -------
    tensorflow.keras.Model
        Compiled CNN model for binding site prediction

    Notes
    -----
    The model architecture is based on DeepBind [1] but simplified for this project.

    References
    ----------
    .. [1] Alipanahi, B., Delong, A., Weirauch, M. T., & Frey, B. J. (2015).
       Predicting the sequence specificities of DNA-and RNA-binding proteins
       by deep learning. Nature biotechnology, 33(8), 831-838.

    Examples
    --------
    >>> model = create_binding_cnn(sequence_length=200)
    >>> model.summary()
    >>> # Train model
    >>> history = model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))
    """
    import tensorflow as tf

    # Input layer for one-hot encoded DNA
    inputs = tf.keras.layers.Input(shape=(sequence_length, 4))

    # Build convolutional layers
    x = inputs
    for filters, kernel_size, pool_size in zip(num_filters, kernel_sizes, pool_sizes):
        x = tf.keras.layers.Conv1D(filters=filters, kernel_size=kernel_size,
                                  activation='relu', padding='same')(x)
        x = tf.keras.layers.MaxPooling1D(pool_size=pool_size)(x)
        x = tf.keras.layers.Dropout(dropout_rate)(x)

    # Flatten and dense layers
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(dense_units, activation='relu')(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)

    # Output layer
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    # Create and compile model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )

    return model
```

**Sphinx Configuration:**

```bash
# Initialize Sphinx in docs/ directory
sphinx-quickstart docs --sep -p "TF Binding Prediction" -a "Team Members" -v "0.1" -l "en"

# Add extensions to docs/conf.py
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.mathjax',
    'sphinx.ext.intersphinx',
]

# Generate API documentation
sphinx-apidoc -o docs/api src

# Add to docs/index.rst
.. toctree::
   :maxdepth: 2
   :caption: Contents

# Build HTML documentation
cd docs
make html
```
