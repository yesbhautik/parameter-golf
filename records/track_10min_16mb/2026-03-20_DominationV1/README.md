# DominationV1: 11L Int6 + Per-dim SmearGate + RoPE50K + SWA50

**Mean val_bpb: 1.1391** (3 seeds verified) | **15.38 MB** | 8xH100 SXM, 600s

## Key Techniques

1. **11 layers**, 512 dim, 8 heads, 4 KV heads (GQA), MLP 3x (hidden=1536)
2. **Per-dimension SmearGate**: learned `sigmoid(Parameter(dim))` gate blending current and previous token embeddings. Each embedding dimension gets its own blend ratio (512 params).
3. **BigramHash (2048x128)**: hashed bigram token-pair context with learned projection to model dim.
4. **Int6 per-row quantization** on MLP + attention weights, int8 for embedding. zstd-22 compression.
5. **RoPE base 50K**: improved position encoding for 2048 seq len.
6. **Muon optimizer** with WD=0.04, momentum 0.99 (warmup 0.92->0.99 over 1500 steps), LR=0.02.
7. **SWA every 50 steps** during last 50% of training (~30 checkpoint average).
8. **Orthogonal init + muP scaling**: orthogonal weight init, output projections scaled by 1/sqrt(2*num_layers).
9. **U-Net skip connections** with learnable per-dim skip weights.
10. **Sliding window eval** stride=64 for near-full context scoring.

## Results (3 seeds, 8xH100 SXM)

| Seed | Steps | ms/step | Post-Q val_bpb | Artifact |
|------|-------|---------|----------------|----------|
| **42** | **8,426** | **71.2** | **1.13781** | **15.38 MB** |
| 1337 | 8,426 | 71.2 | 1.13915 | 15.42 MB |
| 7 | 8,426 | 71.0 | 1.14038 | 15.39 MB |

**Mean: 1.13911** | Range: 0.00257

## Architecture

- 11 layers, 512 dim, MLP 3x (1536 hidden), relu-squared activation
- Vocab 1024 (SentencePiece BPE), seq len 2048, tied embeddings
- RoPE with base 50000, logit softcapping 30.0
- 26.8M parameters

## Run command

```bash
RUN_ID=domv1_s42 SEED=42 \
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=2048 BIGRAM_DIM=128 \
MATRIX_LR=0.02 SCALAR_LR=0.02 TIED_EMBED_LR=0.030 \
MUON_WD=0.04 WEIGHT_DECAY=0.04 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3000 \
ROPE_BASE=50000 \
SWA_ENABLED=1 SWA_EVERY=50 SWA_START_FRAC=0.5 \
MIXED_QUANT_INT6_CATS=mlp,attn \
TRAIN_BATCH_TOKENS=524288 TRAIN_SEQ_LEN=2048 \
MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 EVAL_BATCH_SEQS=32 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```
