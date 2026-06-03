# RDT2 Code Map

Source inspected: remote `ssh jianan@183.230.224.121 -p 50210`, repo `/home/jianan/code/RDT2`, branch `dev_jianan`, conda env `rdt2`.

Local model/data root used by the remote project: `/data/jianan/rdt2/`.

Remote git status at inspection time:
- `dev_jianan...origin/dev_jianan`
- untracked `async_inference_bak/`

## High-Level Purpose

RDT2 contains two related policy paths:

- `RDT2-VQ`: Qwen2.5-VL based autoregressive VLA. It predicts discrete action tokens, then `vqvae` decodes tokens into continuous UMI actions.
- `RDT2-FM`: an RDT action expert trained with a flow-matching objective. It uses Qwen2.5-VL visual-language KV cache as conditioning and predicts continuous action chunks with lower inference latency.

Both paths are built around UMI-style bimanual data: two 384x384 RGB camera views, state/action dim 20, action horizon 24, and 30 Hz control.

## Important Local Paths

- `/data/jianan/rdt2/RDT2-VQ`: local VQ VLA checkpoint.
- `/data/jianan/rdt2/RDT2-FM`: local flow-matching RDT checkpoint.
- `/data/jianan/rdt2/Qwen2.5-VL-7B-Instruct`: local Qwen processor/model source used to avoid Hub access.
- `/data/jianan/rdt2/RVQActionTokenizer`: residual VQ action tokenizer.
- `/data/jianan/rdt2/RVQActionTokenizer/umi_normalizer_wo_downsample_indentity_rot.pt`: action normalizer.
- `/data/jianan/rdt2/instructions.json`: instruction key to natural-language instruction mapping.
- `/data/jianan/rdt2/shard-10-000000`: example shard directory used by local inference scripts.

## Top-Level Directory Map

- `README.md`: upstream usage guide for installation, checkpoints, inference, fine-tuning, and robot deployment.
- `main.py`: argument parser and CLI entry for VQ fine-tuning; calls `train.train(args)`.
- `train.py`: RDT2-VQ training loop around `Qwen2_5_VLForConditionalGeneration`, LoRA/QLoRA options, WebDataset loading, action-token collation, and `VLATrainer`.
- `vla_trainer.py`: Hugging Face trainer customization for VLA training/evaluation.
- `utils.py`: VQ inference/eval helpers. Key functions include `batch_predict_action`, `compute_action_metrics`, and `compute_action_errors`.
- `configs/`: dataset, RDT model, and robot config files.
- `data/`: training/eval dataset code and UMI data utilities.
- `models/`: RDT-FM model, inferencer wrapper, normalizer, and Hugging Face Hub compatibility.
- `vqvae/`: residual/vector-quantized action tokenizer implementation.
- `deploy/`: real robot deployment, calibration, controllers, cameras, collision utilities.
- `examples/`: offline shard inference examples and robot-specific setup notes.
- `async_inference/`: added gRPC async inference service/client wrapping `RDT2Policy`.
- `scripts/`: fine-tuning launch scripts and DeepSpeed configs.
- `requirements/`: extra dependencies for UR5e and Franka deployments.
- `rdt/`: older/alternate RDT training/sample package; separate from the main RDT2-FM wrapper in `models/`.
- `assets/`: README/project assets.
- `async_inference_bak/`: untracked backup copy on remote; do not rely on it as canonical code.

## Main Workflows

### RDT2-VQ Fine-Tuning

Entry chain:

1. `main.py` parses training arguments.
2. `train.py` loads processor/model from `args.tokenizer_name` and `args.pretrained_model_name_or_path`.
3. Optional LoRA/QLoRA adapters are configured through PEFT.
4. `MultiVQVAE.from_pretrained(args.vae_name)` provides action token dimensions.
5. `data.utils.get_instructions_and_blended_train_dataset` loads WebDataset shards and instruction JSON.
6. The collate function turns each sample into a Qwen chat template with image + instruction + `<action>` placeholders, then replaces action placeholders with tokenizer IDs derived from `action_token`.
7. Labels mask everything before the assistant response so loss is computed on action generation.
8. `VLATrainer` handles training/evaluation.

Key config:
- `configs/datasets/shard10_rdt2_fm.yaml` points `shards_dir`, `instruction_path`, and `normalizer_path` at `/data/jianan/rdt2`.
- Fine-tuning scripts are under `scripts/finetune_full_param.sh`, `scripts/finetune_lora.sh`, and `scripts/finetune_rdt.sh`.

### RDT2-VQ Offline Inference

Entry chain:

1. `examples/run_inference_rdt2_vq_shard.py`
2. `examples/shard_inference_utils.py` loads a sample image/action/meta from `/data/jianan/rdt2/shard-10-000000`.
3. `AutoProcessor` and `Qwen2_5_VLForConditionalGeneration` load from `/data/jianan/rdt2/RDT2-VQ`.
4. `MultiVQVAE` loads from `/data/jianan/rdt2/RVQActionTokenizer`.
5. `utils.batch_predict_action` concatenates left/right images, builds a chat prompt ending with `<|im_start|>assistant\n<|quad_start|>`, generates action IDs, converts generated tokenizer IDs back to VQ code IDs, decodes with VQ-VAE, and unnormalizes actions.

### RDT2-FM Offline Inference

Entry chain:

1. `examples/run_inference_rdt2_fm_shard.py`
2. `examples/shard_inference_utils.py` loads sample data and local checkpoint paths.
3. `RDTInferencer` in `models/rdt_inferencer.py` loads:
   - RDT policy from `/data/jianan/rdt2/RDT2-FM`
   - Qwen2.5-VL model from `/data/jianan/rdt2/RDT2-VQ`
   - normalizer from `/data/jianan/rdt2/RVQActionTokenizer/...pt`
4. `RDTInferencer.step` builds ordered camera images, encodes image+instruction with Qwen, extracts selected KV cache layers, normalizes state/action space, and calls `RDTRunner.conditional_sample`.
5. `RDTRunner` performs flow-matching Euler integration for `num_inference_timesteps` steps.

### Real Robot RDT2-FM Inference

Entry chain:

1. `deploy/inference_real_fm.py`
2. Loads robot/camera/gripper config from `configs/robots/eval_bimanual_*.yaml`.
3. Creates `BimanualUmiEnv` from `deploy/umi/real_world/bimanual_umi_env.py`.
4. Reads observations, converts real-world UMI observations through `deploy/umi/real_world/real_inference_util.py`.
5. Calls `RDTInferencer.step`.
6. Converts policy action back to TCP space, applies gripper/table/sphere collision post-processing, and submits action chunks to the environment.

Robot configs:
- `configs/robots/eval_bimanual_ur5e_config.yaml`: UR5e robot IPs, Hik camera serials, Zhixing gripper serials, left-right transform, tracker-to-TCP calibration.
- `configs/robots/eval_bimanual_fr3_config.yaml`: Franka server host/ports, Hik camera serials, Zhixing grippers, transforms/calibration.

### Async gRPC Inference

This is a project-specific wrapper around RDT2-FM for decoupling robot observation submission from action streaming.

Files:
- `async_inference/proto/rdt2_async.proto`: gRPC service schema.
- `async_inference/server.py`: `RDT2AsyncService` with health/reset/submit/stream RPCs.
- `async_inference/rdt2_policy.py`: `RDT2Policy` wrapper around `RDTInferencer`; also has `DummyRDT2Policy`.
- `async_inference/codec.py`: JPEG encode/decode and action flatten/unflatten helpers.
- `async_inference/client_sim.py`: random-image simulated client.
- `async_inference/real_piper_client.py`: real robot/client integration path.

Default paths in `async_inference/rdt2_policy.py`:
- `DEFAULT_MODEL_CONFIG = configs/rdt/post_train.yaml`
- `DEFAULT_PRETRAINED_PATH = /data/jianan/rdt2/RDT2-FM`
- `DEFAULT_VLM_PATH = /data/jianan/rdt2/RDT2-VQ`
- `DEFAULT_QWEN_PROCESSOR_PATH = /data/jianan/rdt2/Qwen2.5-VL-7B-Instruct`
- `DEFAULT_NORMALIZER_PATH = /data/jianan/rdt2/RVQActionTokenizer/umi_normalizer_wo_downsample_indentity_rot.pt`

Server behavior:
- Observation queue has `maxsize=1`; only the newest observation is retained.
- Worker decodes JPEGs, validates state/eef pose, calls policy, flattens `[horizon, action_dim]`, and publishes `ActionChunk`.
- `first_timestep = latest_executed_timestep + 1`.
- `--dummy` serves hold-position chunks without loading the model.
- Torch compile is disabled by default; `--compile-model` opts in.

## Config Notes

### `configs/rdt/post_train.yaml`

- `common.action_chunk_size = 24`
- `common.action_dim = 20`
- `common.state_dim = 20`
- `common.num_cameras = 2`
- `common.fps = 30`
- `dataset.camera_names = ["left_stereo", "right_stereo"]`
- `model.selected_layers = [0..13]`, using Qwen KV cache layers.
- `model.rdt.hidden_size = 1024`
- `model.rdt.depth = 14`
- `model.rdt.num_heads = 8`
- `model.rdt.num_kv_heads = 4`
- `model.noise_scheduler.num_inference_timesteps = 5`

### `configs/bimanual_video_data.yaml`

Defines UMI observation/action shape metadata:
- Cameras: `camera0_rgb`, `camera1_rgb`, raw 480x480, resized 384x384.
- Robot state per arm: eef position 3, rotation 6D, gripper width 1.
- Two arms produce state/action dim 20.
- Action horizon 24.
- Pose representation: relative observation/action.
- Validation ratio defaults to 0.
- Dataset can apply JPEG compression.

## Data Modules

- `data/utils.py`: WebDataset training shard loader and dataset blending. Expects `shard-*.tar` containing `image.jpg`, `action_token.npy`, and `meta.json`.
- `data/umi_video_dataset.py`: zarr/LMDB UMI video dataset for evaluation and conversion-style access. Handles replay buffer loading, validation split, RGB resizing, pose representation, instruction mapping, and normalizer use.
- `data/base_dataset.py`: common dataset base class.
- `data/image_corrupt.py`: optional image corruption/augmentation for VQ training.
- `data/umi/common/replay_buffer.py`: zarr-backed episode buffer.
- `data/umi/common/sampler.py`: sequence sampling and validation masks.
- `data/umi/common/pose_repr_util.py`, `data/umi/pose_util.py`: pose matrix, 6D rotation, 10D pose, axis-angle conversions and geometry losses.
- `data/umi/common/cv2_util.py`: image transforms.

## Model Modules

- `models/rdt_inferencer.py`: inference wrapper for RDT2-FM. Handles loading Qwen VLM, loading `RDTRunner`, language/image encoding, KV-cache extraction, state normalization, action sampling, and reset/cache behavior.
- `models/rdt_runner.py`: `RDTRunner` module. Wraps `models.rdt.model.RDT`, condition adapters, flow-matching timestep sampling, `conditional_sample`, and training loss.
- `models/rdt/model.py`: core RDT transformer.
- `models/rdt/attention.py`, `blocks.py`, `norm.py`, `pos_emb.py`: transformer attention, block, RMS/norm, and positional embedding implementation.
- `models/normalizer/normalizer.py`: `LinearNormalizer` and normalizer serialization.
- `models/hub_mixin.py`: Hugging Face compatible PyTorch model loading/saving helper.
- `vqvae/models/multivqvae.py`: combines position, rotation, and gripper tokenizers.
- `vqvae/models/rvq.py`, `vq.py`, `vqvae.py`: residual VQ and VQ-VAE implementations.
- `vqvae/models/cnn/`: CNN blocks/backbone for tokenizer models.

## Deploy Modules

- `deploy/inference_real_fm.py`: main real-world RDT2-FM deployment script.
- `deploy/vllm_utils.py`: VLM-related deployment utilities.
- `deploy/collision_utils.py`: collision avoidance utilities.
- `deploy/reset_robot_gripper.py`: robot/gripper reset helper.
- `deploy/get_camera_serials.py`, `deploy/get_gripper_serials.py`: hardware discovery helpers.
- `deploy/calibration/`: tracker, Franka/UR calibration scripts, and calibration matrix computation.
- `deploy/umi/real_world/bimanual_umi_env.py`: real-world bimanual environment coordinating cameras, arms, grippers, shared memory, and control timing.
- `deploy/umi/real_world/camera/`: Hik/MVS camera wrappers and multi-camera capture.
- `deploy/umi/real_world/rtde_interpolation_controller.py`: UR RTDE interpolated controller.
- `deploy/umi/real_world/franka_interpolation_controller.py`: Franka interpolated controller.
- `deploy/umi/real_world/zhixing_controller.py`, `zhixing_driver.py`: Zhixing gripper control.
- `deploy/umi/real_world/real_inference_util.py`: real observation/action conversion between environment and policy spaces.
- `deploy/umi/common/`: precise sleep, interpolation, timestamp accumulation.
- `deploy/umi/shared_memory/`: shared memory queue/ring buffer/ndarray utilities.

## Examples And Docs

- `examples/run_inference_rdt2_vq_shard.py`: local shard test for VQ model.
- `examples/run_inference_rdt2_fm_shard.py`: local shard test for FM model.
- `examples/shard_inference_utils.py`: central local path constants, shard sample loader, instruction resolver, action/token reports, CUDA memory reporting.
- `examples/DEPLOYMENT_TIPS.md`: deployment notes.
- `examples/fr3/README.md`, `examples/fr3/launch_franka_server.py`: Franka setup/server notes.
- `examples/ur5e/README.md`: UR5e setup notes.
- `examples/instructions.json`: sample instruction mapping.

## Quick Commands

Offline RDT2-FM shard inference on remote:

```bash
conda activate rdt2
cd /home/jianan/code/RDT2
python examples/run_inference_rdt2_fm_shard.py --sample-id 77016 --device cuda:0
```

Offline RDT2-VQ shard inference on remote:

```bash
conda activate rdt2
cd /home/jianan/code/RDT2
python examples/run_inference_rdt2_vq_shard.py --sample-id 77016 --device cuda:0
```

Async dummy server/client smoke test:

```bash
conda activate rdt2
cd /home/jianan/code/RDT2
python -m async_inference.server --host 0.0.0.0 --port 18080 --dummy
python -m async_inference.client_sim --server 127.0.0.1:18080
```

Async real model server:

```bash
conda activate rdt2
cd /home/jianan/code/RDT2
python -m async_inference.server --host 0.0.0.0 --port 18080 --device cuda:0
```

## Where To Look First

- Need VQ training behavior: `main.py`, `train.py`, `vla_trainer.py`, `data/utils.py`.
- Need VQ inference behavior: `utils.batch_predict_action`, `examples/run_inference_rdt2_vq_shard.py`.
- Need FM inference behavior: `models/rdt_inferencer.py`, `models/rdt_runner.py`, `configs/rdt/post_train.yaml`.
- Need real robot behavior: `deploy/inference_real_fm.py`, `deploy/umi/real_world/bimanual_umi_env.py`, `deploy/umi/real_world/real_inference_util.py`.
- Need async inference behavior: `async_inference/server.py`, `async_inference/rdt2_policy.py`, `async_inference/proto/rdt2_async.proto`.
- Need dataset shape/action semantics: `configs/bimanual_video_data.yaml`, `data/umi_video_dataset.py`, `data/umi/pose_util.py`.
- Need checkpoint/data paths: `examples/shard_inference_utils.py`, `async_inference/rdt2_policy.py`, `configs/datasets/shard10_rdt2_fm.yaml`.

