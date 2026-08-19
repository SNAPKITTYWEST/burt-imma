# Experiment Protocol 001: MMEP Arithmetic Ablation

**Project:** BURT-IMMA  
**Contact:** jessica@collectivekitty.com  
**License:** BSL-1.1

---

## 1. Hypothesis

**Primary hypothesis:** MMEP (Matrix-Memory Equilibrium Propagation) with matrix memory outperforms standard backpropagation on arithmetic reasoning tasks.

**Secondary hypotheses:**
- H2: Matrix memory (C_global + C_expert) is necessary for strong arithmetic performance
- H3: Spectral and entropy constraints improve convergence stability without hurting accuracy
- H4: Sparse MoE routing enables specialization (e.g., one expert for addition, another for multiplication)
- H5: The equilibrium propagation training signal is at least as effective as backpropagation for this domain

---

## 2. Task Description

**Arithmetic reasoning:** Given an arithmetic expression as input, predict the correct result.

**Expression types:**
- Addition: `a + b = ?` (integers, 0-999)
- Subtraction: `a - b = ?` (integers, result >= 0)
- Multiplication: `a * b = ?` (integers, 0-99)
- Division: `a / b = ?` (exact division only, result is integer)
- Chained: `a + b * c - d = ?` (2-4 operations, PEMDAS)
- Nested: `(a + b) * (c - d) = ?` (parentheses)

**Data generation:** `scripts/generate_arithmetic.py`

```python
# generate_arithmetic.py produces:
# - train.jsonl: 100,000 expressions
# - val.jsonl: 10,000 expressions
# - test.jsonl: 10,000 expressions
# Format: {"input": "23 + 45 = ?", "output": "68", "type": "addition", "difficulty": 1}
```

**Difficulty levels:**
1. Single operation, small numbers (0-99)
2. Single operation, larger numbers (0-999)
3. Two operations, small numbers
4. Two operations, larger numbers
5. Three+ operations with parentheses

---

## 3. Experiment Configurations

### 3.1 Configuration Table

| # | Config Name | Description | Key Modification |
|---|-------------|-------------|------------------|
| 1 | `full_mmep` | Complete BURT-IMMA system | None (baseline) |
| 2 | `no_memory` | Remove all matrix memory | C_global = 0, C_expert = 0, no CIFG |
| 3 | `no_constraint` | Remove spectral + entropy constraints | lambda_max = inf, entropy_bound = inf |
| 4 | `no_moe` | Single expert, no routing | num_experts = 1, no router |
| 5 | `standard_bp` | Standard backpropagation training | Replace EP with Adam + backprop |
| 6 | `high_T_free` | Extended free phase | T_free = 100 (vs default 20) |
| 7 | `low_beta` | Very weak nudging | beta = 0.001 (vs default 0.1) |
| 8 | `large_memory` | Doubled memory capacity | d_model = 1024 (vs default 512) |

### 3.2 Detailed Configuration Specifications

#### Config 1: `full_mmep` (Baseline)

```yaml
model:
  d_model: 512
  num_layers: 4
  num_heads: 8
  num_experts: 4
  top_k: 2
  d_expert: 256

training:
  method: mmep
  T_free: 20
  T_nudge: 4
  beta: 0.1
  lr: 1e-4
  epochs: 50
  batch_size: 8

constraints:
  lambda_max: 0.95
  entropy_bound: 0.20
  spectral_projection: true
  entropy_projection: true

memory:
  C_global_dim: 512
  C_expert_dim: 256
  cifg_enabled: true
  forget_bias_init: 0.0
```

#### Config 2: `no_memory`

```yaml
# Same as full_mmep except:
memory:
  C_global_dim: 0       # No global memory
  C_expert_dim: 0       # No expert memory
  cifg_enabled: false   # No CIFG gating
  # Forward pass skips memory retrieval step
```

#### Config 3: `no_constraint`

```yaml
# Same as full_mmep except:
constraints:
  lambda_max: 1000.0       # Effectively no spectral bound
  entropy_bound: 100.0     # Effectively no entropy bound
  spectral_projection: false
  entropy_projection: false
```

#### Config 4: `no_moe`

```yaml
# Same as full_mmep except:
model:
  num_experts: 1    # Single expert
  top_k: 1          # Must route to the only expert
  # No router, no load balancing, no sparsity penalty
```

#### Config 5: `standard_bp`

```yaml
# Same model architecture as full_mmep, but:
training:
  method: backprop    # Standard backpropagation
  optimizer: adam
  lr: 3e-4
  weight_decay: 0.01
  epochs: 50
  batch_size: 32      # Larger batch (no EP memory overhead)
  # No T_free, T_nudge, or beta
```

#### Config 6: `high_T_free`

```yaml
# Same as full_mmep except:
training:
  T_free: 100    # 5x longer free phase (default: 20)
  # Tests whether longer equilibration improves quality
```

#### Config 7: `low_beta`

```yaml
# Same as full_mmep except:
training:
  beta: 0.001    # 100x weaker nudging (default: 0.1)
  # Tests biological plausibility limit
  # Lower beta = more accurate gradient but weaker signal
```

#### Config 8: `large_memory`

```yaml
# Same as full_mmep except:
model:
  d_model: 1024       # 2x model dimension
  d_expert: 512       # 2x expert dimension
memory:
  C_global_dim: 1024  # 4x memory capacity (1024^2 vs 512^2)
  C_expert_dim: 512   # 4x expert memory
```

---

## 4. Metrics

### 4.1 Primary Metrics

| Metric | Definition | Collection Frequency |
|--------|-----------|---------------------|
| **Accuracy** | Exact match of predicted vs. true answer | Every epoch |
| **Convergence speed** | Epoch at which accuracy first exceeds 80% | Once per run |
| **Final accuracy** | Test set accuracy at epoch 50 | End of training |

### 4.2 Secondary Metrics

| Metric | Definition | Collection Frequency |
|--------|-----------|---------------------|
| **Routing entropy** | H(alpha) averaged over test set | Every epoch |
| **Max spectral norm** | max_W(sigma_max(W)) | Every epoch |
| **Memory utilization** | effective_rank(C_global) / d_model | Every 5 epochs |
| **Energy trajectory** | E(t) over training (EP configs only) | Every step |
| **Expert specialization** | Accuracy per expert per operation type | End of training |
| **Loss curve** | Training loss per epoch | Every epoch |

### 4.3 Per-Operation Breakdown

Report accuracy separately for each operation type:
- Addition accuracy
- Subtraction accuracy
- Multiplication accuracy
- Division accuracy
- Chained operations accuracy
- Nested operations accuracy

---

## 5. Falsification Criteria

The MMEP theory makes specific, testable claims. The following criteria define falsification:

### 5.1 Primary Falsification

**If `standard_bp` (Config 5) achieves higher final accuracy than `full_mmep` (Config 1) by more than 2% (with p < 0.05), then:**

The claim that "MMEP outperforms backpropagation on reasoning tasks" is falsified for arithmetic reasoning.

### 5.2 Secondary Falsification Criteria

| Criterion | Condition | What it falsifies |
|-----------|-----------|-------------------|
| F1 | `standard_bp` > `full_mmep` + 2% | EP training advantage |
| F2 | `no_memory` >= `full_mmep` | Matrix memory necessity |
| F3 | `no_constraint` > `full_mmep` + 2% | Constraint benefit |
| F4 | Energy non-monotonic for 10+ consecutive steps | Lyapunov guarantee |
| F5 | Spectral norm > lambda_max after projection | Spectral projection correctness |
| F6 | Entropy > 0.20 + 1e-3 after projection | Entropy projection correctness |
| F7 | `no_moe` >= `full_mmep` | MoE specialization benefit |

### 5.3 Non-Falsification Outcomes

If none of the falsification criteria are triggered:
- MMEP is validated for arithmetic reasoning (not universally)
- Results support the theoretical framework but do not prove it
- Generalization to other domains requires separate experiments

### 5.4 Interpretation Guide

| Outcome | Interpretation | Action |
|---------|---------------|--------|
| All criteria pass | Theory supported | Proceed to harder tasks |
| F1 triggered | EP training not beneficial here | Investigate task properties |
| F2 triggered | Memory unnecessary for arithmetic | Reduce architecture |
| F3 triggered | Constraints too restrictive | Relax bounds |
| F4 triggered | Implementation bug or theory gap | Debug convergence |
| F5 or F6 triggered | Projection implementation error | Fix kernel bugs |
| F7 triggered | MoE overhead not justified | Simplify to single expert |

---

## 6. Hardware Requirements

### 6.1 Minimum Hardware

- **GPU:** NVIDIA RTX 3080 (10 GB VRAM) or equivalent
- **CPU:** 8 cores (for data loading)
- **RAM:** 32 GB
- **Storage:** 20 GB free (checkpoints + logs)
- **CUDA:** 11.8 or later

### 6.2 VRAM Budget per Configuration

| Config | Model Params | Memory Matrices | Activations | Total VRAM |
|--------|-------------|-----------------|-------------|-----------|
| `full_mmep` | 200 MB | 4 MB | 500 MB | ~2 GB |
| `no_memory` | 195 MB | 0 MB | 450 MB | ~1.5 GB |
| `no_constraint` | 200 MB | 4 MB | 500 MB | ~2 GB |
| `no_moe` | 150 MB | 2 MB | 400 MB | ~1.5 GB |
| `standard_bp` | 200 MB | 4 MB | 300 MB | ~1.5 GB |
| `high_T_free` | 200 MB | 4 MB | 500 MB | ~2 GB |
| `low_beta` | 200 MB | 4 MB | 500 MB | ~2 GB |
| `large_memory` | 400 MB | 16 MB | 900 MB | ~3.5 GB |

All configurations fit within RTX 3080's 10 GB with ample headroom.

---

## 7. Expected Runtime

### 7.1 Per-Configuration Runtime

| Config | Time per Epoch | Total (50 epochs) | Notes |
|--------|---------------|-------------------|-------|
| `full_mmep` | 3 min | 2.5 hours | EP overhead: T_free + T_nudge passes |
| `no_memory` | 2.5 min | 2 hours | No memory ops |
| `no_constraint` | 2.5 min | 2 hours | No projection |
| `no_moe` | 2 min | 1.7 hours | No routing/dispatch |
| `standard_bp` | 1 min | 50 min | Standard training (no EP) |
| `high_T_free` | 10 min | 8.3 hours | 5x free phase |
| `low_beta` | 3 min | 2.5 hours | Same as full_mmep |
| `large_memory` | 6 min | 5 hours | 2x compute |

### 7.2 Total Suite Runtime

```
Sequential (worst case): 2.5 + 2 + 2 + 1.7 + 0.8 + 8.3 + 2.5 + 5 = 24.8 hours
With 3 seeds each: 24.8 * 3 = 74.4 hours

Excluding high_T_free: (24.8 - 8.3) * 3 = 49.5 hours
Parallel (2 configs on RTX 3080): ~37 hours

Target: ~4 hours for single-seed pass (excluding high_T_free)
   = 2.5 + 2 + 2 + 1.7 + 0.8 + 2.5 + 5 = 16.5 hours single seed all configs
   
Achievable in ~4 hours: Run full_mmep + standard_bp + no_memory + no_moe (key comparisons)
   = 2.5 + 0.8 + 2 + 1.7 = 7 hours... 

Revised target for "core suite" (4 hours):
   - full_mmep: 2.5h
   - standard_bp: 0.8h  
   - no_memory: limited to 15 epochs = 0.75h
   Total core: ~4 hours
```

### 7.3 Recommended Execution Order

1. `standard_bp` (fastest, establishes backprop baseline)
2. `full_mmep` (primary comparison)
3. `no_memory` (test memory contribution)
4. `no_moe` (test routing contribution)
5. `no_constraint` (test constraint contribution)
6. `low_beta` (test nudging sensitivity)
7. `large_memory` (test scaling)
8. `high_T_free` (test convergence theory, run overnight)

---

## 8. Data

### 8.1 Data Generation

```bash
python scripts/generate_arithmetic.py \
    --train_size 100000 \
    --val_size 10000 \
    --test_size 10000 \
    --max_value 999 \
    --max_ops 4 \
    --seed 42 \
    --output_dir data/arithmetic/
```

### 8.2 Data Format

```json
{"input": "23 + 45 = ?", "output": "68", "type": "addition", "difficulty": 1, "num_ops": 1}
{"input": "156 * 7 = ?", "output": "1092", "type": "multiplication", "difficulty": 2, "num_ops": 1}
{"input": "(12 + 8) * 3 = ?", "output": "60", "type": "nested", "difficulty": 3, "num_ops": 2}
```

### 8.3 Data Statistics

| Split | Size | Difficulty Distribution | Operation Distribution |
|-------|------|------------------------|----------------------|
| Train | 100K | D1: 30%, D2: 25%, D3: 20%, D4: 15%, D5: 10% | +: 25%, -: 25%, *: 20%, /: 10%, chain: 12%, nest: 8% |
| Val | 10K | Same as train | Same as train |
| Test | 10K | Same as train | Same as train |

### 8.4 Tokenization

- Character-level tokenization (digits, operators, parentheses, spaces)
- Vocabulary size: 20 tokens (`0-9`, `+`, `-`, `*`, `/`, `(`, `)`, `=`, `?`, ` `, `<PAD>`)
- Maximum sequence length: 32 (sufficient for all expressions)

---

## 9. Execution Script

```bash
#!/bin/bash
# run_experiment_001.sh

CONFIGS="full_mmep standard_bp no_memory no_constraint no_moe low_beta large_memory high_T_free"
SEEDS="42 137 256"
DATA_DIR="data/arithmetic"
OUTPUT_DIR="results/experiment_001"

# Generate data (if not exists)
if [ ! -f "$DATA_DIR/train.jsonl" ]; then
    python scripts/generate_arithmetic.py --output_dir $DATA_DIR --seed 42
fi

# Run experiments
for config in $CONFIGS; do
    for seed in $SEEDS; do
        echo "Running config=$config seed=$seed"
        python scripts/run_training.py \
            --config configs/experiment_001/${config}.yaml \
            --seed $seed \
            --data_dir $DATA_DIR \
            --output_dir $OUTPUT_DIR/${config}/seed_${seed} \
            --epochs 50 \
            --eval_every 1 \
            --checkpoint_every 10
    done
done

# Generate report
python scripts/analyze_results.py \
    --results_dir $OUTPUT_DIR \
    --output results/experiment_001_report.json
```

---

## 10. Analysis Plan

### 10.1 Statistical Tests

1. **Paired t-test:** Compare each config against `full_mmep` on final accuracy (3 seeds each)
2. **Bonferroni correction:** Adjust p-values for 7 comparisons (alpha = 0.05/7 = 0.0071)
3. **Effect size:** Report Cohen's d for each comparison
4. **Confidence intervals:** 95% CI on accuracy difference

### 10.2 Visualization

- Learning curves (accuracy vs. epoch) for all configs
- Energy trajectories (EP configs only)
- Expert specialization heatmap (operation type vs. expert)
- Spectral norm evolution over training
- Entropy evolution over training

### 10.3 Report Format

```json
{
    "experiment": "001_mmep_arithmetic_ablation",
    "date": "YYYY-MM-DD",
    "hardware": "RTX 3080, 10GB",
    "results": {
        "full_mmep": {"accuracy": [s1, s2, s3], "mean": X, "std": Y},
        "standard_bp": {"accuracy": [s1, s2, s3], "mean": X, "std": Y},
        ...
    },
    "falsification": {
        "F1": {"triggered": false, "p_value": 0.XX, "effect_size": X.XX},
        ...
    },
    "conclusion": "Theory supported|falsified for criterion FN"
}
```

---

## 11. Reproducibility Checklist

- [ ] All random seeds documented and fixed
- [ ] Data generation script deterministic
- [ ] Model initialization deterministic
- [ ] CUDA deterministic mode enabled (`torch.backends.cudnn.deterministic = True`)
- [ ] All hyperparameters documented in config YAML
- [ ] PyTorch version recorded
- [ ] CUDA version recorded
- [ ] GPU model recorded
- [ ] Training logs saved (loss per step)
- [ ] Checkpoints saved at epoch 10, 20, 30, 40, 50
- [ ] Environment frozen (`pip freeze > requirements_experiment_001.txt`)
