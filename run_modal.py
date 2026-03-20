"""
Modal deployment script for parameter-golf training on 8xH100 SXM.

Usage:
    Val-only run:     modal run run_modal.py --mode valonly --profile counter --seeds 1337,42,7
    Standard run:     modal run run_modal.py --mode standard --profile counter --seeds 1337,42,7
    One-off sweep:    modal run run_modal.py --mode standard --profile counter --seed 7 --quant-mode int5mlp --num-layers 10 --muon-wd 0.045 --tag wd045

This will:
    - Download the FineWeb dataset inside Modal (cached in a Volume)
    - Run torchrun with 8xH100 GPUs for 10 minutes
    - Download train.log and model artifacts to your local machine
"""

import modal

app = modal.App("parameter-golf")

data_vol = modal.Volume.from_name("parameter-golf-data", create_if_missing=True)
output_vol = modal.Volume.from_name("parameter-golf-output", create_if_missing=True)

TRAIN_SCRIPT_STANDARD = "records/track_10min_16mb/2026-03-19_StandardOptimal/train_gpt.py"
TRAIN_SCRIPT_VALONLY = "records/track_10min_16mb/2026-03-19_CombinedOptimal/train_gpt.py"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.10",
        "numpy",
        "sentencepiece",
        "huggingface-hub",
        "tqdm",
        "setuptools",
        "typing-extensions==4.15.0",
        "zstandard",
        extra_options="--extra-index-url https://download.pytorch.org/whl/cu124",
    )
    .add_local_file("data/cached_challenge_fineweb.py", "/root/data/cached_challenge_fineweb.py")
    .add_local_file(TRAIN_SCRIPT_STANDARD, "/root/train_gpt_standard.py")
    .add_local_file(TRAIN_SCRIPT_VALONLY, "/root/train_gpt_valonly.py")
)


@app.function(
    image=image,
    gpu="H100:8",
    timeout=45 * 60,
    volumes={
        "/data": data_vol,
        "/output": output_vol,
    },
)
def train(
    mode: str = "valonly",
    seed: int = 1337,
    tag: str = "",
    profile: str = "counter",
    num_layers: int = 0,
    quant_mode: str = "auto",
    muon_wd: float = -1.0,
    weight_decay: float = -1.0,
    swa_every: int = 0,
    extra_env: str = "",
):
    import os
    import shutil
    import subprocess
    import sys

    def parse_extra_env(raw: str) -> dict[str, str]:
        out = {}
        if not raw:
            return out
        for item in raw.split(","):
            item = item.strip()
            if not item:
                continue
            if "=" not in item:
                raise ValueError(f"Invalid extra_env entry (expected KEY=VALUE): {item}")
            k, v = item.split("=", 1)
            out[k.strip()] = v.strip()
        return out

    def profile_env(selected_mode: str, selected_profile: str) -> dict[str, str]:
        if selected_mode == "standard":
            if selected_profile in {"counter", "counter_v7"}:
                return {
                    "VAL_LOSS_EVERY": "500",
                    "TRAIN_LOG_EVERY": "100",
                    "TRAIN_SEQ_LEN": "2048",
                    "TRAIN_BATCH_TOKENS": "786432",
                    "NUM_LAYERS": "10",
                    "SWA_ENABLED": "1",
                    "SWA_EVERY": "50",
                    "SWA_START_FRAC": "0.5",
                    "STE_QAT_ENABLED": "0",
                    "MIXED_QUANT_INT6_CATS": "mlp,attn",
                    "INT6_QUANT_RANGE_MLP": "15",
                    "INT6_QUANT_RANGE_ATTN": "31",
                    "INT6_QUANT_RANGE_OTHER": "31",
                    "FP16_PASSTHROUGH_PATTERNS": "tok_emb,blocks.9.attn.c_k",
                    "WEIGHT_DECAY": "0.04",
                    "MUON_WD": "0.04",
                }
            if selected_profile == "baseline":
                return {
                    "VAL_LOSS_EVERY": "500",
                    "TRAIN_LOG_EVERY": "100",
                    "TRAIN_SEQ_LEN": "2048",
                    "TRAIN_BATCH_TOKENS": "786432",
                    "NUM_LAYERS": "9",
                    "SWA_ENABLED": "1",
                    "SWA_EVERY": "50",
                    "SWA_START_FRAC": "0.5",
                    "STE_QAT_ENABLED": "0",
                    "MIXED_QUANT_INT6_CATS": "mlp,attn",
                    "INT6_QUANT_RANGE_MLP": "31",
                    "INT6_QUANT_RANGE_ATTN": "31",
                    "INT6_QUANT_RANGE_OTHER": "31",
                    "FP16_PASSTHROUGH_PATTERNS": "tok_emb,blocks.8.attn.c_k",
                    "WEIGHT_DECAY": "0.04",
                    "MUON_WD": "0.04",
                }
        else:
            if selected_profile in {"counter", "counter_v7"}:
                return {
                    "VAL_LOSS_EVERY": "500",
                    "TRAIN_LOG_EVERY": "100",
                    "TRAIN_SEQ_LEN": "1024",
                    "TRAIN_BATCH_TOKENS": "524288",
                    "NUM_LAYERS": "11",
                    "MATRIX_LR": "0.025",
                    "SCALAR_LR": "0.025",
                    "SWA_ENABLED": "0",
                    "STE_QAT_ENABLED": "0",
                    "MIXED_QUANT_INT6_CATS": "mlp,attn",
                    "FP16_PASSTHROUGH_PATTERNS": "tok_emb,blocks.10.attn.c_k",
                    "WEIGHT_DECAY": "0.034",
                    "MUON_WD": "0.034",
                }
            if selected_profile == "baseline":
                return {
                    "VAL_LOSS_EVERY": "1000",
                    "TRAIN_LOG_EVERY": "200",
                    "TRAIN_SEQ_LEN": "2048",
                    "TRAIN_BATCH_TOKENS": "786432",
                    "NUM_LAYERS": "9",
                    "MATRIX_LR": "0.020",
                    "SCALAR_LR": "0.020",
                    "SWA_ENABLED": "0",
                    "STE_QAT_ENABLED": "1",
                    "MIXED_QUANT_INT6_CATS": "mlp,attn,other",
                    "FP16_PASSTHROUGH_PATTERNS": "tok_emb",
                    "WEIGHT_DECAY": "0.01",
                    "MUON_WD": "0.02",
                }
        raise ValueError(f"Unknown profile '{selected_profile}' for mode '{selected_mode}'")

    os.chdir("/root")

    data_base = "/data/datasets/fineweb10B_sp1024"
    tokenizer_dir = "/data/tokenizers"
    val_shard = f"{data_base}/fineweb_val_000000.bin"
    is_standard = mode == "standard"
    train_shards = "80" if is_standard else "1"
    mode_prefix = "standard" if is_standard else "valonly"
    run_id_base = f"{mode_prefix}_{profile}_s{seed}"
    run_id = f"{run_id_base}_{tag}" if tag else run_id_base

    # ----------------------------------------------------------------
    # Step 1: Download data
    # ----------------------------------------------------------------
    need_download = not os.path.exists(val_shard)
    if is_standard:
        need_download = need_download or not os.path.exists(f"{data_base}/fineweb_train_000079.bin")

    if need_download:
        print(f"=== Downloading FineWeb data ({train_shards} train shards) ===", flush=True)
        subprocess.run(
            [
                sys.executable,
                "/root/data/cached_challenge_fineweb.py",
                "--variant", "sp1024",
                "--train-shards", train_shards,
            ],
            check=True,
            env={**os.environ, "PYTHONPATH": "/root"},
            cwd="/root",
        )
        local_ds = "/root/data/datasets/fineweb10B_sp1024"
        local_tok = "/root/data/tokenizers"
        os.makedirs(data_base, exist_ok=True)
        os.makedirs(tokenizer_dir, exist_ok=True)
        for f in os.listdir(local_ds):
            src = os.path.join(local_ds, f)
            dst = os.path.join(data_base, f)
            if not os.path.exists(dst):
                shutil.copy2(src, dst)
        for f in os.listdir(local_tok):
            src = os.path.join(local_tok, f)
            dst = os.path.join(tokenizer_dir, f)
            if not os.path.exists(dst):
                shutil.copy2(src, dst)
        data_vol.commit()
        print("=== Data cached to volume ===", flush=True)
    else:
        print("=== Data already cached ===", flush=True)

    # ----------------------------------------------------------------
    # Step 2: Determine data path
    # ----------------------------------------------------------------
    if is_standard:
        data_path = data_base
    else:
        valonly_dir = "/data/datasets/fineweb10B_sp1024_valonly"
        os.makedirs(valonly_dir, exist_ok=True)
        valonly_train = f"{valonly_dir}/fineweb_train_000000.bin"
        valonly_val = f"{valonly_dir}/fineweb_val_000000.bin"
        if not os.path.exists(valonly_train):
            shutil.copy2(val_shard, valonly_train)
        if not os.path.exists(valonly_val):
            shutil.copy2(val_shard, valonly_val)
        data_vol.commit()
        data_path = valonly_dir

    # ----------------------------------------------------------------
    # Step 3: Run training with torchrun on 8xH100
    # ----------------------------------------------------------------
    print(f"=== Starting training (mode={mode}, profile={profile}, data={data_path}) ===", flush=True)

    env = {
        **os.environ,
        "RUN_ID": run_id,
        "SEED": str(seed),
        "DATA_PATH": data_path,
        "TOKENIZER_PATH": f"{tokenizer_dir}/fineweb_1024_bpe.model",
        "VOCAB_SIZE": "1024",
        "MAX_WALLCLOCK_SECONDS": "600",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    }
    env.update(profile_env(mode, profile))
    if quant_mode == "int5mlp":
        env.update(
            {
                "MIXED_QUANT_INT6_CATS": "mlp,attn",
                "INT6_QUANT_RANGE_MLP": "15",
                "INT6_QUANT_RANGE_ATTN": "31",
                "INT6_QUANT_RANGE_OTHER": "31",
                "STE_QAT_ENABLED": "0",
            }
        )
    elif quant_mode == "int6all":
        env.update(
            {
                "MIXED_QUANT_INT6_CATS": "mlp,attn,other",
                "INT6_QUANT_RANGE_MLP": "31",
                "INT6_QUANT_RANGE_ATTN": "31",
                "INT6_QUANT_RANGE_OTHER": "31",
            }
        )
    elif quant_mode == "int6mlp_attn":
        env.update(
            {
                "MIXED_QUANT_INT6_CATS": "mlp,attn",
                "INT6_QUANT_RANGE_MLP": "31",
                "INT6_QUANT_RANGE_ATTN": "31",
                "INT6_QUANT_RANGE_OTHER": "31",
            }
        )
    elif quant_mode != "auto":
        raise ValueError(f"Unknown quant_mode: {quant_mode}")
    if num_layers > 0:
        env["NUM_LAYERS"] = str(num_layers)
    if muon_wd >= 0:
        env["MUON_WD"] = str(muon_wd)
    if weight_decay >= 0:
        env["WEIGHT_DECAY"] = str(weight_decay)
    if swa_every > 0:
        env["SWA_ENABLED"] = "1"
        env["SWA_EVERY"] = str(swa_every)
    env.update(parse_extra_env(extra_env))
    train_script = "/root/train_gpt_standard.py" if is_standard else "/root/train_gpt_valonly.py"

    if not is_standard:
        # Keep val-only data path isolated.
        env["DATA_PATH"] = data_path
    else:
        env["DATA_PATH"] = data_base

    result = subprocess.run(
        [
            sys.executable, "-m", "torch.distributed.run",
            "--standalone",
            "--nproc_per_node=8",
            train_script,
        ],
        env=env,
        cwd="/root",
        stdout=sys.stdout,
        stderr=sys.stderr,
    )

    print(f"\n=== Training finished with exit code {result.returncode} ===", flush=True)

    # ----------------------------------------------------------------
    # Step 4: Copy outputs to the output volume
    # ----------------------------------------------------------------
    output_base = f"/output/{run_id}"
    os.makedirs(output_base, exist_ok=True)

    for fname in ["final_model.pt", "final_model.int8.ptz"]:
        src = f"/root/{fname}"
        if os.path.exists(src):
            shutil.copy2(src, f"{output_base}/{fname}")
            print(f"  Saved {fname} ({os.path.getsize(src)} bytes)", flush=True)

    log_dir = "/root/logs"
    if os.path.isdir(log_dir):
        for fname in os.listdir(log_dir):
            src = os.path.join(log_dir, fname)
            shutil.copy2(src, f"{output_base}/{fname}")
            print(f"  Saved log: {fname}", flush=True)

    output_vol.commit()
    print(f"\n=== All outputs saved to volume 'parameter-golf-output' at /{run_id}/ ===")


@app.function(
    image=modal.Image.debian_slim(),
    volumes={"/output": output_vol},
)
def download_results(run_id: str = "valonly_counter_s1337"):
    import os

    output_base = f"/output/{run_id}"
    if not os.path.isdir(output_base):
        print(f"No results found for run_id={run_id}.")
        return

    for fname in sorted(os.listdir(output_base)):
        fpath = os.path.join(output_base, fname)
        size = os.path.getsize(fpath)
        print(f"  {fname}: {size:,} bytes")

        if fname.endswith(".txt"):
            print(f"\n--- {fname} contents (last 30 lines) ---")
            with open(fpath) as f:
                lines = f.readlines()
            for line in lines[-30:]:
                print(line, end="")
            print(f"\n--- end {fname} ---\n")


@app.local_entrypoint()
def main(
    mode: str = "valonly",
    profile: str = "counter",
    seed: int = 1337,
    seeds: str = "",
    tag: str = "",
    num_layers: int = 0,
    quant_mode: str = "auto",
    muon_wd: float = -1.0,
    weight_decay: float = -1.0,
    swa_every: int = 0,
    extra_env: str = "",
):
    seed_list = [int(s.strip()) for s in seeds.split(",") if s.strip()] if seeds else [seed]
    print(
        f"Launching training on Modal 8xH100 SXM (mode={mode}, profile={profile}, "
        f"seeds={seed_list}, tag={tag}, quant_mode={quant_mode})..."
    )
    print("This will take ~15 minutes (10 min train + ~5 min eval + overhead)\n")
    for i, run_seed in enumerate(seed_list):
        run_tag = tag
        if len(seed_list) > 1 and not run_tag:
            run_tag = f"batch{i+1}"
        mode_prefix = "standard" if mode == "standard" else "valonly"
        run_id_base = f"{mode_prefix}_{profile}_s{run_seed}"
        run_id = f"{run_id_base}_{run_tag}" if run_tag else run_id_base
        train.remote(
            mode=mode,
            seed=run_seed,
            tag=run_tag,
            profile=profile,
            num_layers=num_layers,
            quant_mode=quant_mode,
            muon_wd=muon_wd,
            weight_decay=weight_decay,
            swa_every=swa_every,
            extra_env=extra_env,
        )
        print(f"\n=== Fetching results for {run_id} ===\n")
        download_results.remote(run_id=run_id)
        print(f"\nTo download files locally for {run_id}:")
        print(f"  modal volume get parameter-golf-output {run_id}/{run_id}.txt ./train.log")
        print(f"  modal volume get parameter-golf-output {run_id}/final_model.int8.ptz ./final_model.int8.ptz")
