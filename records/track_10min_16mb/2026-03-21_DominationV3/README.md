# DominationV3: 11L EMA + Partial RoPE + LN Scale + TTT(25ep)

**Mean val_bpb: 1.12604** (3 seeds) | **Best: 1.12561** | 8xH100 SXM

## Key Techniques

1. **11-layer GPT** with 512 model dim, 8 heads, 4 KV heads, MLP 3x.
2. **Partial RoPE** (16/64 dims): RoPE on 25% of head dims; rest position-free.
3. **LN Scale** (`1/sqrt(layer_idx+1)`): Damp deeper layer norm outputs.
4. **EMA averaging** (decay=0.997).
5. **BigramHash(4096x128)** for local context.
6. **Mixed int6 quantization** on `mlp`, `attn`, and `tok_emb` + zstd-22.
7. **25-epoch aggressive SGD TTT** (lr=0.01, momentum=0.9, ALL blocks unfrozen) on already-graded tokens.
8. **XSA disabled** to save ~1.4ms/step and gain ~130 more training steps.
9. **Sliding-window evaluation** (stride=64).

## Compliance

- Trains only on `fineweb_train_*` shards (80 shards).
- TTT runs at eval time on the quantized model, adapting only to tokens already scored.
- Training capped to 599.8s. TTT ~388s + sliding eval ~197s = ~585s total eval (under 10 min).
- All artifacts under 16,000,000 bytes.

## Results (3 seeds, 8xH100 SXM)

| Seed | val_bpb | train_time_ms | ttt_time_ms | total_artifact_bytes |
|------|---------|---------------|-------------|----------------------|
| 1337 | **1.12561480** | 599782 | 387925 | 15801567 |
| 7    | 1.12677932 | 599800 | 388329 | 15778949 |
| 42   | 1.12571883 | 599800 | 388158 | 15989167 |

**Mean:** 1.12603765
**Stddev:** 0.00064441

## Repro

```bash
modal run records/track_10min_16mb/2026-03-21_DominationV3/run_modal.py \
  --mode standard --profile domv3 --seed 1337 --bigram-vocab 4096 \
  --extra-env "FP16_PASSTHROUGH_PATTERNS=;MIXED_QUANT_INT6_CATS=mlp,attn,tok_emb;MAX_WALLCLOCK_SECONDS=599.8;XSA_LAST_N=0;ROPE_DIMS=16;LN_SCALE=1;TTT_ENABLED=1;TTT_LR=0.01;TTT_MOMENTUM=0.9;TTT_EPOCHS=25;TTT_FREEZE_BLOCKS=0"
```
