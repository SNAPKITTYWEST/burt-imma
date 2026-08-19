# AlexNet-to-BURT Bridge: Vision Backbone Integration

**Project:** BURT-IMMA  
**Contact:** jessica@collectivekitty.com  
**License:** BSL-1.1

---

## 1. Overview

This document specifies the integration bridge between AlexNet's convolutional feature extraction and BURT's Boolean reasoning layers. The bridge replaces AlexNet's fully-connected classification head with MMEP-trained Boolean reasoning, creating a hybrid architecture that leverages proven vision features while applying principled constraint-based reasoning for classification.

---

## 2. Architecture Mapping

### 2.1 Original AlexNet (Krizhevsky 2012)

```
Input (224x224x3)
  -> Conv1 (96 filters, 11x11, stride 4) + ReLU + MaxPool + LRN
  -> Conv2 (256 filters, 5x5, pad 2) + ReLU + MaxPool + LRN
  -> Conv3 (384 filters, 3x3, pad 1) + ReLU
  -> Conv4 (384 filters, 3x3, pad 1) + ReLU
  -> Conv5 (256 filters, 3x3, pad 1) + ReLU + MaxPool
  -> Flatten (9216)
  -> FC6 (4096) + ReLU + Dropout
  -> FC7 (4096) + ReLU + Dropout
  -> FC8 (1000) + Softmax
```

### 2.2 BURT-AlexNet Bridge

```
Input (224x224x3)
  -> [AlexNet Conv Backbone: Conv1-Conv5] (feature extraction, retained)
  -> Flatten (9216)
  -> Linear Projection (9216 -> d_model=512)
  -> [BURT Reasoning Block x 2] (replaces FC6, FC7, FC8)
      -> GatesNorm + SmoothLeaky
      -> CIFGMatrixMemory (retrieval)
      -> GatesRouter -> Expert MoE
      -> BooleanConstraintLayer
      -> QuantumInterferenceResolver
  -> Constrained Classification Head (d_model -> num_classes)
  -> Constrained Softmax (entropy <= 0.20)
```

### 2.3 Layer Correspondence

| AlexNet Layer | BURT-AlexNet | Change |
|---------------|--------------|--------|
| Conv1 | Conv1 | Retained (frozen or fine-tuned) |
| Conv2 | Conv2 | Retained (frozen or fine-tuned) |
| Conv3 | Conv3 | Retained (frozen or fine-tuned) |
| Conv4 | Conv4 | Retained (frozen or fine-tuned) |
| Conv5 | Conv5 | Retained (frozen or fine-tuned) |
| FC6 (4096) | BURT Block 1 | Replaced with MMEP-trained block |
| FC7 (4096) | BURT Block 2 | Replaced with MMEP-trained block |
| FC8 (1000) | Constrained Head | Replaced with entropy-bounded head |
| ReLU | SmoothLeaky | Smooth activation for EP compatibility |
| Dropout | GatesNorm + Sparsity | Structural regularization |
| LRN | Spectral Normalization | Modern normalization |
| Softmax | Constrained Softmax | Entropy bounded at 0.20 |

---

## 3. Feature Extraction (Conv Backbone)

### 3.1 Retained Architecture

The convolutional layers (Conv1-Conv5) are retained from AlexNet as the feature extraction backbone. These layers have been proven effective for learning hierarchical visual features:

- **Conv1:** Edge detectors, Gabor-like filters
- **Conv2:** Texture and color detectors
- **Conv3:** Part detectors
- **Conv4:** Object part combinations
- **Conv5:** Whole-object representations

### 3.2 Modifications to Conv Backbone

| Modification | Original | Bridge | Rationale |
|-------------|----------|--------|-----------|
| Activation | ReLU | SmoothLeaky | Required for EP energy landscape smoothness |
| Normalization | LRN | Spectral Norm | Bounded singular values for convergence |
| Pooling | MaxPool | MaxPool | Retained (compatible with EP) |
| Initialization | Random | ImageNet pretrained | Transfer learning |

### 3.3 Fine-tuning Strategy

```
Phase 1 (memorization): Conv backbone frozen, only projection trained
Phase 2 (query):        Conv backbone frozen, BURT blocks trained
Phase 3 (convergence):  Conv backbone unfrozen with 10x lower LR
Phase 4 (ablation):     All layers trainable
```

Learning rates:
- Conv layers: `lr_base * 0.1` (slow adaptation)
- Projection: `lr_base` (full learning rate)
- BURT blocks: `lr_base` (full learning rate)
- Classification head: `lr_base * 2.0` (faster convergence)

---

## 4. Replacing FC Layers with MMEP-Trained Boolean Reasoning

### 4.1 Projection Layer

The flattened convolutional output (9216 dims) is projected to the BURT model dimension:

```python
class ConvToBURT(nn.Module):
    def __init__(self, conv_dim=9216, d_model=512):
        self.projection = nn.Linear(conv_dim, d_model)
        self.norm = GatesNormalization(d_model)
        self.activation = SmoothLeaky(k=1.0)
    
    def forward(self, conv_features):
        x = self.projection(conv_features.flatten(1))
        x = self.norm(x)
        x = self.activation(x)
        return x  # [batch, d_model]
```

### 4.2 BURT Reasoning Blocks

Two BURT blocks replace FC6 and FC7:

**Block 1 (replaces FC6):** Feature integration + memory retrieval
- Retrieves relevant stored patterns from C_global
- Routes to domain-specific experts (animal vs. vehicle vs. scene, etc.)
- Applies Boolean constraints to ensure logical consistency

**Block 2 (replaces FC7):** Abstract reasoning + classification prep
- Combines expert outputs via interference
- Enforces Huntington postulates on classification features
- Prepares logits for constrained output

### 4.3 Constrained Classification Head

```python
class ConstrainedClassificationHead(nn.Module):
    def __init__(self, d_model=512, num_classes=1000, entropy_bound=0.20):
        self.linear = SpectralNorm(nn.Linear(d_model, num_classes))
        self.entropy_bound = entropy_bound
    
    def forward(self, x):
        logits = self.linear(x)
        probs = constrained_softmax(logits, entropy_bound=self.entropy_bound)
        return logits, probs
```

The entropy bound of 0.20 on the classification distribution means the model must be confident in its predictions. A uniform distribution over 1000 classes would have entropy ~6.9 nats; bounding at 0.20 forces the model to concentrate probability mass on very few classes.

---

## 5. Data Augmentation: BiEncoder Retrieval Replaces Color Jitter

### 5.1 Original AlexNet Augmentation

Krizhevsky 2012 used:
- Random 224x224 crops from 256x256 images
- Horizontal flips
- **Color jitter via PCA** (fancy PCA augmentation on RGB channels)

### 5.2 BURT-AlexNet Augmentation

The color jitter augmentation is replaced by BiEncoder retrieval augmentation:

```python
class BiEncoderAugmentation:
    def __init__(self, memory_bank, k=5):
        self.memory_bank = memory_bank  # Pre-computed feature bank
        self.k = k  # Number of retrieved neighbors
    
    def augment(self, x, features):
        # Retrieve k nearest neighbors from memory bank
        distances, indices = self.memory_bank.search(features, k=self.k)
        neighbors = self.memory_bank.get_images(indices)
        
        # Create augmented batch: original + retrieved context
        augmented = {
            'image': x,
            'context_features': self.memory_bank.get_features(indices),
            'context_labels': self.memory_bank.get_labels(indices),
            'distances': distances
        }
        return augmented
```

**Rationale:** Rather than perturbing color channels (which adds noise), BiEncoder retrieval provides semantically relevant context from the training set. This context is stored in the matrix memory and used during inference, enabling the model to reason about an image by comparing it to similar stored examples.

### 5.3 Retained Augmentations

- Random 224x224 crops (retained)
- Horizontal flips (retained)
- Random erasing (added for robustness)

---

## 6. Training Schedule (Adapted from Krizhevsky 2012)

### 6.1 Original Schedule

Krizhevsky trained for 90 epochs with:
- SGD with momentum 0.9
- Weight decay 0.0005
- LR starts at 0.01, divided by 10 when validation error plateaus
- Batch size 128 across 2 GPUs

### 6.2 BURT-AlexNet Schedule

Adapted for single-GPU MMEP training:

| Phase | Epochs | LR (conv) | LR (BURT) | Method | Batch |
|-------|--------|-----------|------------|--------|-------|
| 1: Memorize | 5 | 0 (frozen) | 1e-3 | Forward only | 32 |
| 2: Route | 20 | 0 (frozen) | 3e-4 | Backprop | 16 |
| 3: Converge | 50 | 1e-5 | 1e-4 | MMEP (EP) | 8 |
| 4: Fine-tune | 15 | 1e-5 | 5e-5 | MMEP (EP) | 8 |

**Total: 90 epochs** (matching Krizhevsky's total training duration)

### 6.3 LR Decay

```python
# Cosine annealing with warm restarts
scheduler = CosineAnnealingWarmRestarts(
    optimizer,
    T_0=10,       # First restart period
    T_mult=2,     # Double period after each restart
    eta_min=1e-6  # Minimum LR
)
```

### 6.4 Weight Decay

- Conv layers: 0.0005 (matching Krizhevsky)
- BURT layers: 0.0 (spectral norm + sparsity replaces weight decay)
- Classification head: 0.0001

---

## 7. Entropy Bound on Classification Head

### 7.1 Constraint Definition

The output distribution after softmax must satisfy:

```
H(p) = -sum(p_i * log(p_i)) <= 0.20
```

For ImageNet (1000 classes), this means:
- Maximum entropy of uniform distribution: log(1000) = 6.91 nats
- Bound of 0.20 nats implies effective support of ~1.22 classes
- The model must be very confident, concentrating > 80% probability on the top class

### 7.2 Enforcement

During training:
```python
def constrained_classification_loss(logits, targets, entropy_bound=0.20):
    probs = softmax(logits)
    ce_loss = cross_entropy(logits, targets)
    
    # Entropy penalty
    entropy = -sum(probs * log(probs + 1e-10), dim=-1)
    entropy_penalty = relu(entropy - entropy_bound).mean()
    
    return ce_loss + 10.0 * entropy_penalty
```

During inference:
```python
def constrained_predict(logits, entropy_bound=0.20):
    # Temperature scaling to meet entropy bound
    temperature = find_temperature(logits, entropy_bound)
    probs = softmax(logits / temperature)
    return probs
```

### 7.3 Implications

- Forces the model to learn discriminative features
- Prevents hedging across similar classes
- Ensures interpretable predictions (few classes with high probability)
- Acts as implicit confidence calibration

---

## 8. Spectral Normalization on All Weight Matrices

### 8.1 Application

Every weight matrix in the BURT-AlexNet bridge is spectrally normalized:

```python
def apply_spectral_normalization(model, lambda_max=0.95):
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # Constrain largest singular value
            spectral_norm(module, name='weight', n_power_iterations=1)
            
            # Additional: clip to lambda_max after normalization
            with torch.no_grad():
                U, S, V = torch.svd(module.weight_orig)
                S_clipped = torch.clamp(S, max=lambda_max)
                module.weight_orig.copy_(U @ torch.diag(S_clipped) @ V.T)
```

### 8.2 Where Applied

| Layer Type | Spectral Bound | Update Frequency |
|-----------|---------------|-----------------|
| Conv1-Conv5 | 0.95 | Every step (if unfrozen) |
| Projection | 0.95 | Every step |
| BURT attention (Q, K, V) | 0.95 | Every step |
| Expert weights | 0.95 | Every step |
| Router weights | 0.95 | Every step |
| Classification head | 0.95 | Every step |
| Memory gate weights | 0.95 | Every step |

### 8.3 Interaction with EP Convergence

Spectral normalization is essential for EP convergence in the BURT blocks:
- Guarantees contraction mapping in free phase
- Ensures unique equilibrium for each input
- Prevents activation explosion during nudged phase
- See [MMEP_THEORY.md](./MMEP_THEORY.md) Section 7 for the Lyapunov proof

---

## 9. Complete Forward Pass

```python
class BURTAlexNet(nn.Module):
    def forward(self, x, memory_state=None):
        # === Conv Backbone (from AlexNet) ===
        x = self.conv1(x)           # [B, 96, 55, 55]
        x = self.smooth_leaky(x)
        x = self.pool1(x)           # [B, 96, 27, 27]
        
        x = self.conv2(x)           # [B, 256, 27, 27]
        x = self.smooth_leaky(x)
        x = self.pool2(x)           # [B, 256, 13, 13]
        
        x = self.conv3(x)           # [B, 384, 13, 13]
        x = self.smooth_leaky(x)
        
        x = self.conv4(x)           # [B, 384, 13, 13]
        x = self.smooth_leaky(x)
        
        x = self.conv5(x)           # [B, 256, 13, 13]
        x = self.smooth_leaky(x)
        x = self.pool5(x)           # [B, 256, 6, 6]
        
        # === Bridge: Conv -> BURT ===
        x = x.flatten(1)            # [B, 9216]
        x = self.projection(x)     # [B, d_model=512]
        x = self.gates_norm_proj(x)
        x = self.smooth_leaky(x)
        
        # === BURT Block 1 (replaces FC6) ===
        # Memory retrieval
        if memory_state is None:
            memory_state = self.C_global
        mem_context = memory_state @ x.unsqueeze(-1)  # [B, d_model, 1]
        x = x + mem_context.squeeze(-1)
        
        # Expert routing
        expert_ids, weights = self.router1(x)
        expert_outs = [self.experts1[k](x) for k in expert_ids]
        x = x + self.resolver1(expert_outs, weights)
        
        # Boolean constraint
        x = x + self.boolean1(self.gates_norm1(x))
        
        # Memory update
        self.update_memory(x)
        
        # === BURT Block 2 (replaces FC7) ===
        mem_context = memory_state @ x.unsqueeze(-1)
        x = x + mem_context.squeeze(-1)
        
        expert_ids, weights = self.router2(x)
        expert_outs = [self.experts2[k](x) for k in expert_ids]
        x = x + self.resolver2(expert_outs, weights)
        
        x = x + self.boolean2(self.gates_norm2(x))
        
        # === Classification Head (replaces FC8) ===
        logits = self.classification_head(x)  # [B, num_classes]
        probs = constrained_softmax(logits, entropy_bound=0.20)
        
        return logits, probs, memory_state
```

---

## 10. Expected Performance

### 10.1 ImageNet Comparison

| Model | Top-1 Acc | Top-5 Acc | Parameters |
|-------|-----------|-----------|-----------|
| AlexNet (2012) | 63.3% | 84.6% | 61M |
| AlexNet + BURT (projected) | 68-72% | 87-90% | 45M |

The BURT bridge reduces parameters (no 4096-dim FC layers) while improving accuracy through structured reasoning and memory retrieval.

### 10.2 Reasoning Tasks

On visual reasoning benchmarks (CLEVR, visual arithmetic):
- Expected improvement over baseline AlexNet: 15-25%
- Boolean constraint enforcement ensures logically consistent answers
- Memory retrieval enables few-shot pattern matching

---

## 11. References

- Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). "ImageNet Classification with Deep Convolutional Neural Networks." NeurIPS.
- [MMEP_THEORY.md](./MMEP_THEORY.md) - Training algorithm
- [BURT_SPEC.md](./BURT_SPEC.md) - Full BURT architecture specification
- [kernel_api.md](./kernel_api.md) - CUDA implementations for constrained_softmax
