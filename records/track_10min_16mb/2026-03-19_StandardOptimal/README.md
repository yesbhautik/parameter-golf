## Standard Optimal

Standard FineWeb training (no val-only memorization) with best-in-class architecture and optimization.

### Techniques

1. **MLP 3x expansion (h=1536)**: Wider feedforward for increased capacity (~21.8M params).

2. **STE fake-int6 quantization during training**: All `CastedLinear` weights are fake-quantized to int6 ([-31, 31]) via Straight-Through Estimator, reducing post-training quantization penalty to ~0.001 BPB.

3. **Mixed post-training quantization**: int6 per-row (31 levels) for STE-protected block weights, int8 per-row (127 levels) for embedding.

4. **Sliding window evaluation (stride=64)**: Every scored token gets 960+ tokens of context.

5. **Tuned Muon optimizer**: Momentum 0.99, lower LR (0.02/0.02/0.03), warmdown 3000 steps, seq_len=4096.

### Configuration

- Layout: `VOCAB_SIZE=1024 NUM_LAYERS=9 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=3`
- Tied embeddings: `TIE_EMBEDDINGS=1`
- Learning rates: `MATRIX_LR=0.02 SCALAR_LR=0.02 TIED_EMBED_LR=0.03`
- Optimizer: `MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 MUON_MOMENTUM_WARMUP_STEPS=1500`
- Batching: `TRAIN_BATCH_TOKENS=393216 TRAIN_SEQ_LEN=4096`
- Eval: `EVAL_STRIDE=64 EVAL_BATCH_SEQS=64`

### Command (8xH100)

```bash
python3 data/cached_challenge_fineweb.py --variant sp1024

RUN_ID=standard_optimal \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
MAX_WALLCLOCK_SECONDS=600 \
VAL_LOSS_EVERY=1000 \
TRAIN_LOG_EVERY=200 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```
