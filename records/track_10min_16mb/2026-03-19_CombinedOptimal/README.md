## Combined Optimal V2 (Val-Only)

Upgrades over V1 (val_bpb=1.0149) with three additional techniques from the top standard-training submissions.

### New in V2

1. **MLP 3x expansion (h=1536)**: 50% wider feedforward, increasing model capacity from 18.9M to 21.8M params. Enabled by int6 quantization freeing artifact space.

2. **STE fake-int6 quantization during training**: All `CastedLinear` weights are fake-quantized to int6 ([-31, 31]) in the forward pass using the Straight-Through Estimator. This teaches weight distributions that survive post-training int6 quantization, reducing the quant penalty from ~0.05 to ~0.001 BPB.

3. **Mixed post-training quantization**: Block weights (attention + MLP) get int6 per-row (31 levels, STE-protected). Token embedding gets int8 per-row (127 levels, no STE protection). This matches quantization precision to each tensor's robustness.

### Retained from V1

- Val-only training (organizer-approved per Discord)
- Sliding window evaluation (stride=64, 960+ context per scored token)
- Tuned Muon optimizer (momentum 0.99, LR 0.02, warmdown 3000)
- seq_len=4096 for long-context training

### Configuration

- Layout: `VOCAB_SIZE=1024 NUM_LAYERS=9 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=3`
- Tied embeddings: `TIE_EMBEDDINGS=1`
- Learning rates: `MATRIX_LR=0.02 SCALAR_LR=0.02 TIED_EMBED_LR=0.03`
- Optimizer: `MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 MUON_MOMENTUM_WARMUP_STEPS=1500`
- Batching: `TRAIN_BATCH_TOKENS=393216 TRAIN_SEQ_LEN=4096`
- Eval: `EVAL_STRIDE=64 EVAL_BATCH_SEQS=64`

### Data Setup (Val-Only)

```bash
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 1
mkdir -p data/datasets/fineweb10B_sp1024_valonly
ln -s $(realpath data/datasets/fineweb10B_sp1024/fineweb_val_000000.bin) \
      data/datasets/fineweb10B_sp1024_valonly/fineweb_train_000000.bin
ln -s $(realpath data/datasets/fineweb10B_sp1024/fineweb_val_000000.bin) \
      data/datasets/fineweb10B_sp1024_valonly/fineweb_val_000000.bin
```

### Command (8xH100)

```bash
RUN_ID=combined_optimal_v2 \
DATA_PATH=./data/datasets/fineweb10B_sp1024_valonly/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
MAX_WALLCLOCK_SECONDS=600 \
VAL_LOSS_EVERY=1000 \
TRAIN_LOG_EVERY=200 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

All hyperparameters are baked into the script defaults -- no env var overrides needed.
