# BELKA-BERT: Molecular Property Prediction with BERT

A comprehensive framework for predicting molecular binding properties using fine-tuned BERT models. This repository contains implementations of BERT-based approaches for predicting protein target binding (BRD4, HSA, sEH) using chemical SMILES representations and building block information.

## Overview

BELKA-BERT leverages pre-trained language models (specifically ChemBERTa) adapted for molecular property prediction. The framework implements multiple training strategies including:

- **Domain-specific MLM pretraining** on BELKA dataset molecules
- **Hybrid models** combining SMILES embeddings with molecular fingerprints
- **Fine-tuning approaches** with and without masked language modeling (MLM)
- **Systematic ablation studies** isolating contributions of each component
- **Multi-target prediction** with three protein targets: BRD4, HSA, and sEH

## Datasets

The project uses the following parquet files:

- `train_multilabel_combined.parquet` - Main training dataset with multilabel annotations
- `belka_agg_downsampled.parquet` - Aggregated and downsampled dataset for hybrid models
- `train_filtered.parquet` - Filtered training data

**Key columns:**

- `molecule_smiles` - SMILES representation of molecules
- `buildingblock1_smiles`, `buildingblock2_smiles`, `buildingblock3_smiles` - Building block components
- `BRD4`, `HSA`, `sEH` - Binary labels for protein binding targets

## Repository Structure

### Notebooks

#### 1. **BELKA_BERT_NoMLM.ipynb** (638 lines)

Standard BERT fine-tuning approach without masked language modeling.

**Key features:**

- Data loading and SMILES validation using RDKit
- Stratified train/validation/test split
- Configurable hyperparameters (epochs, batch size, learning rates)
- Classification heads on top of frozen/unfrozen encoder
- Evaluation metrics: F1, ROC-AUC, Average Precision
- Confusion matrix and attention heatmap visualizations

**Model configuration:**

```
- Base model: seyonec/ChemBERTa-zinc-base-v1
- Max length: 128 tokens
- Batch size: 32
- Learning rate (head): 2e-3, (encoder): 5e-5
- Epochs: 5
```

#### 2. **BELKABERT-No-FineTuning.ipynb** (683 lines)

Transfer learning approach using frozen BERT embeddings without fine-tuning.

**Key features:**

- Uses pre-trained ChemBERTa embeddings without updating encoder weights
- Data downsampling with pattern-based balancing
- Lightweight classification heads for fast training
- Comprehensive performance metrics and confusion matrices
- Attention visualization and interpretation

#### 3. **MLM_BelkaBERT.ipynb** (1667 lines)

Advanced domain-adaptive pretraining with masked language modeling.

**Key features:**

- Domain-specific MLM pretraining on BELKA molecules
- DuckDB-based efficient data aggregation and downsampling
- Masked language modeling with configurable MLM probability (default: 15%)
- Multi-stage fine-tuning (head + encoder)
- t-SNE visualization of embeddings
- SMILES vs domain-adapted model comparisons
- Few-shot learning performance analysis

**Configuration:**

```
- MLM epochs: 2, Pretraining LR: 5e-5
- Fine-tune epochs: 3 (stage 1), 3 (stage 2)
- Batch size: 64 (pretraining), 32 (fine-tuning)
```

#### 4. **BELKA-BERT-Hybrid-Final.ipynb** (2552 lines)

State-of-the-art hybrid approach combining multiple modalities.

**Key features:**

- **Hybrid fusion**: SMILES embeddings + molecular fingerprints (2048-bit, radius 2)
- **Leakage-safe splitting** using building block triplets
- **MLM pretraining** on BELKA data
- **Multi-head attention** and learnable fusion mechanism
- **Anti-hallucination regularization** (promiscuity penalty)
- **Precision-based thresholding** for deployment-ready predictions
- **Comprehensive analysis**:
  - Calibration curves
  - Precision-recall curves
  - Enrichment factor analysis
  - Chemical driver identification (which building blocks matter most)
  - Building block sensitivity analysis
  - Label combination co-binding analysis
  - Latent space clustering visualization
  - Model error analysis (molecular weight, lipophilicity correlations)
  - Attention mechanism visualization

**Advanced features:**

- Cross-validation support (5-fold)
- Early stopping with patience
- FP16 mixed precision training
- Output directory management for experiment tracking

#### 5. **BELKA_BERT-Ablation_Redesign.ipynb** (Latest)

Comprehensive ablation study isolating individual component contributions.

**Key features:**

- **Systematic ablations** of 8 variants covering:
  - Full method (reference): MLM + FP + BB + fine-tuning
  - Component removal: w/o_DAPT, w/o_FP, w/o_BB, w/o_FT
  - Loss components: w/o_pos_weight, w/o_promisc_reg
  - Baseline: SMILES_only (MLM without auxiliary inputs)
- **Runtime-aware configuration** with capped MLM steps and eval samples
- **Leakage-safe splitting** using building block triplets (identical to Hybrid-Final)
- **Temperature scaling** and precision-based threshold selection
- **Hallucination metrics** to track spurious multi-target predictions
- **Calibration analysis** (ECE, Brier score)
- **Comparative results**: Delta analysis vs SMILES-only baseline
- **Multi-seed support** for mean±std reporting (default: seed=42)

**Configuration:**

```
- Base model: seyonec/ChemBERTa-zinc-base-v1 or MLM checkpoint
- MLM steps: 5,000 (fixed for runtime consistency)
- MLM sample max: 250,000
- Fine-tune phases: Head-only (1 epoch, ≤1200 steps) + Encoder (3 epochs, ≤1200 steps)
- Batch size: 32
- Two-phase learning rates: lr_head=2e-3, lr_encoder=3e-5
- Anti-hallucination: lambda_promiscuity=0.05, target_precision=0.90
```

**Outputs:**

- `ablation_results.jsonl` - Per-run detailed metrics
- `ablation_results.csv` - Aggregated results with seed averaging
- Component contribution analysis and delta metrics vs baseline

## Key Technologies

- **PyTorch** - Deep learning framework
- **Transformers (Hugging Face)** - Pre-trained BERT models and utilities
- **RDKit** - Cheminformatics library for SMILES validation and fingerprint generation
- **Scikit-learn** - Machine learning utilities and metrics
- **Pandas & NumPy** - Data manipulation
- **DuckDB** - Efficient SQL-based data aggregation
- **Matplotlib** - Visualization
- **tqdm** - Progress tracking

## Training Strategies

### Strategy 1: Standard Fine-tuning (NoMLM)

- Load pre-trained ChemBERTa
- Add classification heads
- Train heads and optionally fine-tune encoder
- Simple, fast, good baseline

### Strategy 2: Frozen Embeddings (No-FineTuning)

- Use ChemBERTa without updating weights
- Train only shallow classification layers
- Minimal computational cost
- Good for resource-constrained scenarios

### Strategy 3: Domain-Adaptive MLM (MLM_BelkaBERT)

- Pre-train on BELKA molecules using masked language modeling
- Adapt BERT to molecular domain
- Then fine-tune classification heads
- Better domain specialization

### Strategy 4: Hybrid Fusion (Hybrid-Final)

- Combine SMILES embeddings with molecular fingerprints
- MLM pretraining on domain data
- Anti-hallucination penalties
- Precision-based decision thresholds
- Best performance and interpretability

## Configuration

Each notebook includes a `CFG` class with configurable hyperparameters:

```python
class CFG:
    data_path = "train_multilabel_combined.parquet"
    smiles_col = "molecule_smiles"
    label_cols = ["BRD4", "HSA", "sEH"]
    model_name = "seyonec/ChemBERTa-zinc-base-v1"

    # Training
    epochs = 5
    train_batch_size = 32
    valid_batch_size = 64
    lr_heads = 2e-3
    lr_encoder = 5e-5
    weight_decay = 1e-4

    # Tokenization
    max_length = 128

    # Data split
    valid_size = 0.15
    test_size = 0.15
```

## Usage

### Basic Setup

1. Install dependencies:

```bash
pip install torch transformers scikit-learn pandas numpy rdkit matplotlib tqdm duckdb
```

2. Prepare data (ensure parquet files are in working directory)

3. Open any notebook in Jupyter and run cells sequentially

### Running a Specific Strategy

Each notebook is self-contained and can be run independently:

1. **For quick baseline:** Use `BELKABERT-No-FineTuning.ipynb`
2. **For standard training:** Use `BELKA_BERT_NoMLM.ipynb`
3. **For domain adaptation:** Use `MLM_BelkaBERT.ipynb`
4. **For best performance:** Use `BELKA-BERT-Hybrid-Final.ipynb`

## Model Performance

The notebooks include comprehensive evaluation:

- **Classification metrics:** F1 score, ROC-AUC, Average Precision, Accuracy
- **Calibration:** Expected Calibration Error, confidence distributions
- **Analysis:** Confusion matrices, attention maps, feature importance
- **Enrichment:** Early recognition performance via enrichment factors
- **Interpretability:** Which building blocks drive predictions

## Key Results

### Hybrid Model Features:

- **Multi-modal fusion:** Combines learned SMILES embeddings with structural fingerprints
- **Leakage-safe:** Splits on building block combinations to prevent data leakage
- **Anti-hallucination:** Regularizes against spurious multi-target predictions
- **Precision-driven:** Thresholds predictions based on target precision
- **Well-calibrated:** Includes calibration curves and confidence analysis

### Ablation Study Findings (BELKA_BERT-Ablation_Redesign):

The ablation analysis reveals component contributions through systematic removal:

- **DAPT/MLM**: Domain-adaptive pretraining improves test AP and calibration
- **Fingerprints (FP)**: Structural information boosts performance, especially for harder targets
- **Building Blocks (BB)**: Provides moderate improvement; less critical than FP
- **Fine-tuning (FT)**: Essential for reaching competitive AP scores
- **Positive weighting**: Handles class imbalance effectively
- **Promiscuity regularization**: Reduces spurious multi-target hallucinations
- **Baseline comparison**: SMILES-only (MLM) establishes reference for delta metrics

Results saved to `ablation_results.csv` with mean±std across seeds and delta analysis vs SMILES-only baseline.

## Output

Models and results are typically saved to:

- `runs_mlm_hybrid_paper/` - Hybrid model outputs (BELKA-BERT-Hybrid-Final)
- `ablation_q1_aligned/` - Ablation study outputs (BELKA_BERT-Ablation_Redesign)
  - `ablation_results.jsonl` - Per-run metrics
  - `ablation_results.csv` - Aggregated results with seed averaging
  - `fp_cache/` - Cached molecular fingerprints
  - `canonicalized.parquet` - Canonicalized SMILES
- `ckpt_mlm_belka/` - MLM pretraining checkpoints
- Various CSV/pickle files with predictions and metrics

## References

- Base model: [ChemBERTa-zinc-base-v1](https://huggingface.co/seyonec/ChemBERTa-zinc-base-v1)
- BERT: Devlin et al., 2019
- Molecular representations: SMILES and Morgan fingerprints

## Notes

- SMILES validation is performed using RDKit to ensure data quality
- Stratified splitting by label combinations prevents skewed train/test distributions
- Multiple random seeds ensure reproducibility
- GPU acceleration is automatically detected and used if available
- The framework supports both CPU and GPU training

## Future Improvements

- Ensemble methods combining multiple models
- Active learning for sample selection
- Additional molecular fingerprint types (ECFP, Avalon)
- Larger scale pre-training on expanded chemical databases
- Real-world deployment with API endpoints
