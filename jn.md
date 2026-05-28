# RDT2-FM shard-10 smoke fine-tune

## Environment

- Host: `jianan@183.230.224.121 -p 50210`
- Code: `/home/jianan/code/RDT2`
- Conda env: `rdt2` via `/data/miniconda3`
- Data/model root: `/data/jianan/rdt2`
- Training shard: `/data/jianan/rdt2/shard-10-000000.tar`
- RDT2-FM checkpoint: `/data/jianan/rdt2/RDT2-FM`
- RDT2-VQ checkpoint: `/data/jianan/rdt2/RDT2-VQ`
- Local Qwen processor: `/data/jianan/rdt2/Qwen2.5-VL-7B-Instruct`
- Dataset config: `configs/datasets/shard10_rdt2_fm.yaml`
- Smoke DeepSpeed config: `scripts/zero1_smoke.json`

## Data verification

Verified with `/data/miniconda3/bin/conda run -n rdt2 python /tmp/verify_shard10.py`:

- Sample prefix: `77016`
- Required files present in tar: `image.jpg`, `action.npy`, `meta.json`
- Image size: `(768, 384)`
- Action shape: `(24, 20)`
- `meta["sub_task_instruction_key"]` found in `/data/jianan/rdt2/instructions.json`
- Instruction preview: `Pour water from the plastic bottle into the cup`

## Successful smoke command

`zero1_smoke.json` keeps ZeRO-1 but lowers all-gather/reduce buckets to `100000000` and disables overlap. This avoided the temporary gradient-reduce OOM seen with the stock `scripts/zero1.json` on 24 GiB GPUs.

```bash
source /data/miniconda3/etc/profile.d/conda.sh
conda activate rdt2
cd /home/jianan/code/RDT2

RUN_DIR=/data/jianan/rdt2/finetune_runs/rdt2_fm_shard10_smoke_$(date +%Y%m%d_%H%M%S)
mkdir -p "$RUN_DIR"

(
  while true; do
    nvidia-smi -i 2,3 \
      --query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu,utilization.memory \
      --format=csv,noheader,nounits
    sleep 2
  done
) > "$RUN_DIR/gpu_usage.csv" &
MONITOR_PID=$!

CUDA_VISIBLE_DEVICES=2,3 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
WANDB_MODE=offline \
WANDB_DIR="$RUN_DIR/wandb" \
WANDB_PROJECT=rdt2-action-expert-shard10-smoke \
PYTHONPATH=/home/jianan/code/RDT2 \
accelerate launch \
  --num_processes=2 \
  --num_machines=1 \
  --mixed_precision=bf16 \
  --main_process_port=29521 \
  rdt/main.py \
  --deepspeed=scripts/zero1_smoke.json \
  --config_path=./configs/rdt/post_train.yaml \
  --pretrained_model_name_or_path=/data/jianan/rdt2/RDT2-FM \
  --pretrained_vision_language_model_name_or_path=/data/jianan/rdt2/RDT2-VQ \
  --processor_name_or_path=/data/jianan/rdt2/Qwen2.5-VL-7B-Instruct \
  --output_dir="$RUN_DIR/checkpoints" \
  --logging_dir="$RUN_DIR/logs" \
  --webdataset_config=configs/datasets/shard10_rdt2_fm.yaml \
  --train_batch_size=1 \
  --sample_batch_size=1 \
  --num_sample_batches=2 \
  --max_train_steps=80 \
  --checkpointing_period=1000 \
  --checkpoints_total_limit=1 \
  --sample_period=10 \
  --lr_scheduler=constant \
  --learning_rate=1e-4 \
  --mixed_precision=bf16 \
  --dataloader_num_workers=2 \
  --use_8bit_adam \
  --allow_tf32 \
  --set_grads_to_none \
  --report_to=wandb \
  2>&1 | tee "$RUN_DIR/train.log"

kill "$MONITOR_PID"
```

## Runs

### 30-step stock ZeRO-1 attempt

- Run dir: `/data/jianan/rdt2/finetune_runs/rdt2_fm_shard10_smoke_20260526_175003`
- Result: failed at step 2 with CUDA OOM during ZeRO-1 gradient reduce.
- Peak process memory was about `22.85 GiB`; the failed allocation was `904 MiB`.

### 30-step smoke ZeRO-1 run

- Run dir: `/data/jianan/rdt2/finetune_runs/rdt2_fm_shard10_smoke_20260526_175207`
- Result: completed 30 steps and saved model.
- Loss was noisy over only 30 steps, so the plan's fallback 80-step check was used.
- Deduplicated first 5 loss mean: `0.014164`
- Deduplicated last 5 loss mean: `0.015826`
- Peak GPU memory: `21470 MiB`
- Average GPU util: `17.17%`; peak GPU util: `100%`

### 80-step final smoke run

- Run dir: `/data/jianan/rdt2/finetune_runs/rdt2_fm_shard10_smoke_20260526_175518`
- Result: completed 80 steps and saved model.
- Deduplicated first 5 losses: `[0.00253, 0.0457, 0.0234, 0.017, 0.0108]`
- Deduplicated last 5 losses: `[0.0179, 0.00333, 0.00317, 0.00194, 0.00665]`
- First 5 mean: `0.019886`
- Last 5 mean: `0.006598`
- Sampling logs were emitted at steps 10, 20, 30, 40, 50, 60, 70, and 80.
- Final sample metrics: `action_mse_error=0.0015389049949590117`, `action_geodesic_error_rot=1.79296875`.
- Peak GPU memory: `21470 MiB`
- Average GPU util: `24.08%`; peak GPU util: `100%`
- Average memory util: `2.39%`; peak memory util: `28%`

## Compared with official `scripts/finetune_rdt.sh`

The official script is a long-run template. The smoke run intentionally differs in these places:

- Dataset config: official uses `configs/datasets/example.yaml`; smoke uses `configs/datasets/shard10_rdt2_fm.yaml` for `/data/jianan/rdt2/shard-10-000000.tar`.
- `PYTHONPATH`: official leaves `<repository-path>` as a placeholder; smoke uses `/home/jianan/code/RDT2`.
- VLM checkpoint: official uses the Hugging Face id `robotics-diffusion-transformer/RDT2-VQ`; smoke uses offline local `/data/jianan/rdt2/RDT2-VQ`.
- Processor: official code hard-codes `Qwen/Qwen2.5-VL-7B-Instruct`; smoke adds `--processor_name_or_path` and uses `/data/jianan/rdt2/Qwen2.5-VL-7B-Instruct` to avoid network access.
- RDT action expert checkpoint: smoke passes `--pretrained_model_name_or_path=/data/jianan/rdt2/RDT2-FM`. The official script does not pass this argument, so the current code would construct the RDT action expert from config instead of loading `RDT2-FM`.
- Batch size: official uses `train_batch_size=64`, `sample_batch_size=32`; smoke uses `1/1` because the goal is only to validate the pipeline on two 24 GiB GPUs.
- Train length: official uses `1000000` steps; smoke used `30` then `80` steps.
- DeepSpeed: official uses `scripts/zero1.json`; smoke uses `scripts/zero1_smoke.json` after stock ZeRO-1 OOMed during gradient reduce on the remote 24 GiB GPUs.
- Image augmentation: official enables `--image_aug`; smoke disables it to keep data/debug behavior minimal. Enable it for a real fine-tune if matching the official recommendation is desired.
- Optimizer memory: smoke adds `--use_8bit_adam`, `--set_grads_to_none`, and `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` for the 24 GiB smoke run.
- Logging/output: official writes under repo-local `outputs/` and `logs/`; smoke writes all artifacts under `/data/jianan/rdt2/finetune_runs/`.
- `CFLAGS`/`LDFLAGS`: official exports them for local compiler/library discovery. The smoke run succeeded without them; keep them if rebuilding native extensions or following the official script exactly.

No code change is needed from this comparison. For future non-smoke fine-tuning, the main decision is whether to keep loading `/data/jianan/rdt2/RDT2-FM` via `--pretrained_model_name_or_path`. If the goal is fine-tuning the released RDT2-FM action expert, keep it; omitting it follows the current official script literally but initializes the action expert from config in this code path.

## Formal shard-10 RDT2-FM training command

This is the recommended command for a real run on the current remote server. It follows the official `scripts/finetune_rdt.sh` RDT2-FM flow, but uses local checkpoints/data, loads `/data/jianan/rdt2/RDT2-FM`, enables the local processor path, keeps `--image_aug`, and uses the smoke-verified ZeRO-1 small-bucket config for the 24 GiB GPUs.

```bash
source /data/miniconda3/etc/profile.d/conda.sh
conda activate rdt2
cd /home/jianan/code/RDT2

MAX_TRAIN_STEPS=1000000
RUN_NAME=rdt2_fm_shard10_formal
RUN_DIR=/data/jianan/rdt2/finetune_runs/${RUN_NAME}_$(date +%Y%m%d_%H%M%S)
mkdir -p "$RUN_DIR"

(
  while true; do
    nvidia-smi -i 2,3 \
      --query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu,utilization.memory \
      --format=csv,noheader,nounits
    sleep 2
  done
) > "$RUN_DIR/gpu_usage.csv" &
MONITOR_PID=$!

CUDA_VISIBLE_DEVICES=2,3 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
WANDB_MODE=offline \
WANDB_DIR="$RUN_DIR/wandb" \
WANDB_PROJECT=rdt2-action-expert-shard10-formal \
PYTHONPATH=/home/jianan/code/RDT2 \
accelerate launch \
  --num_processes=2 \
  --num_machines=1 \
  --mixed_precision=bf16 \
  --main_process_port=29531 \
  rdt/main.py \
  --deepspeed=scripts/zero1_smoke.json \
  --config_path=./configs/rdt/post_train.yaml \
  --pretrained_model_name_or_path=/data/jianan/rdt2/RDT2-FM \
  --pretrained_vision_language_model_name_or_path=/data/jianan/rdt2/RDT2-VQ \
  --processor_name_or_path=/data/jianan/rdt2/Qwen2.5-VL-7B-Instruct \
  --output_dir="$RUN_DIR/checkpoints" \
  --logging_dir="$RUN_DIR/logs" \
  --webdataset_config=configs/datasets/shard10_rdt2_fm.yaml \
  --train_batch_size=1 \
  --sample_batch_size=1 \
  --num_sample_batches=2 \
  --max_train_steps="$MAX_TRAIN_STEPS" \
  --checkpointing_period=5000 \
  --checkpoints_total_limit=40 \
  --sample_period=1000 \
  --lr_scheduler=constant \
  --learning_rate=1e-4 \
  --mixed_precision=bf16 \
  --dataloader_num_workers=2 \
  --image_aug \
  --use_8bit_adam \
  --allow_tf32 \
  --set_grads_to_none \
  --report_to=wandb \
  2>&1 | tee "$RUN_DIR/train.log"

kill "$MONITOR_PID"
```

### Formal command short validation

- Test command: `MAX_TRAIN_STEPS=20 RUN_NAME=rdt2_fm_shard10_formal_test /tmp/run_rdt2_fm_shard10_formal.sh`
- Run dir: `/data/jianan/rdt2/finetune_runs/rdt2_fm_shard10_formal_test_20260526_182005`
- Result: completed 20 steps with `--image_aug`, saved model, and exited with `TRAIN_STATUS=0`.
- Deduplicated loss count: `20`
- First 5 losses: `[0.00308, 0.00964, 0.00369, 0.027, 0.0187]`
- Last 5 losses: `[0.00812, 0.0097, 0.0334, 0.0303, 0.0156]`
- Peak GPU memory: `21472 MiB`
- Average GPU util: `11.26%`; peak GPU util: `100%`
- Average memory util: `1.38%`; peak memory util: `21%`
