import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch

from utils import compute_action_errors


DATA_ROOT = Path(os.environ.get("RDT2_DATA_ROOT", "/data/jianan/rdt2"))
SHARD_DIR = Path(os.environ.get("RDT2_SHARD_DIR", str(DATA_ROOT / "shard-10-000000")))
INSTRUCTIONS_PATH = Path(os.environ.get("RDT2_INSTRUCTIONS_PATH", str(DATA_ROOT / "instructions.json")))
RDT2_VQ_PATH = DATA_ROOT / "RDT2-VQ"
RDT2_FM_PATH = DATA_ROOT / "RDT2-FM"
QWEN_PATH = DATA_ROOT / "Qwen2.5-VL-7B-Instruct"
VQVAE_PATH = DATA_ROOT / "RVQActionTokenizer"
NORMALIZER_PATH = VQVAE_PATH / "umi_normalizer_wo_downsample_indentity_rot.pt"


def use_repo_root():
    repo_root = Path(__file__).resolve().parents[1]
    os.chdir(repo_root)


def load_shard_sample(sample_id):
    image_path = SHARD_DIR / f"{sample_id}.image.jpg"
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(image_path)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if image.shape[:2] != (384, 768):
        image = cv2.resize(image, (768, 384), interpolation=cv2.INTER_AREA)

    left_image = image[:, :384].copy()
    right_image = image[:, 384:].copy()

    with open(SHARD_DIR / f"{sample_id}.meta.json", "r") as f:
        meta = json.load(f)

    return {
        "left_image": left_image,
        "right_image": right_image,
        "action": np.load(SHARD_DIR / f"{sample_id}.action.npy"),
        "action_token": np.load(SHARD_DIR / f"{sample_id}.action_token.npy"),
        "meta": meta,
    }


def resolve_instruction(sample, explicit_instruction=None):
    if explicit_instruction is not None:
        return explicit_instruction

    instruction_key = sample["meta"].get("sub_task_instruction_key")
    if instruction_key is None:
        raise KeyError("sample meta has no sub_task_instruction_key")

    with open(INSTRUCTIONS_PATH, "r") as f:
        instructions = json.load(f)
    instruction = instructions.get(instruction_key)
    if instruction is None:
        raise KeyError(f"instruction key not found in {INSTRUCTIONS_PATH}: {instruction_key}")
    return instruction


def print_action_report(name, action_pred, action_gt):
    if isinstance(action_pred, np.ndarray):
        action_pred = torch.from_numpy(action_pred)
    if isinstance(action_gt, np.ndarray):
        action_gt = torch.from_numpy(action_gt)

    pred = action_pred.detach().cpu().float().unsqueeze(0)
    gt = action_gt.detach().cpu().float().unsqueeze(0)
    abs_err = (pred - gt).abs()
    metrics = compute_action_errors(pred, gt, torch.tensor([True]), num_robot=2)

    print(f"{name} action shape: {tuple(pred.shape[1:])}")
    print(f"{name} mean_abs_error: {abs_err.mean().item():.8f}")
    print(f"{name} max_abs_error: {abs_err.max().item():.8f}")
    print(f"{name} action_mse_error: {metrics['action_mse_error'].item():.10f}")
    print(f"{name} action_mse_error_pos: {metrics['action_mse_error_pos'].item():.10f}")
    print(f"{name} action_geodesic_error_rot_deg: {metrics['action_geodesic_error_rot'].item():.6f}")
    print(f"{name} action_mse_error_width: {metrics['action_mse_error_width'].item():.10f}")
    print(f"{name} first_pred_row: {np.array2string(pred[0, 0].numpy(), precision=4, suppress_small=True)}")
    print(f"{name} first_gt_row: {np.array2string(gt[0, 0].numpy(), precision=4, suppress_small=True)}")


def print_token_report(action_tokens_pred, action_tokens_gt):
    pred = action_tokens_pred.detach().cpu().to(torch.long).reshape(-1)
    gt = torch.from_numpy(action_tokens_gt).to(torch.long).reshape(-1)
    n = min(pred.numel(), gt.numel())
    matches = (pred[:n] == gt[:n]).sum().item()

    print(f"RDT2-VQ token shape: pred={tuple(pred.shape)} gt={tuple(gt.shape)}")
    print(f"RDT2-VQ token_match: {matches}/{n}")
    print(f"RDT2-VQ pred_tokens_head: {pred[:10].tolist()}")
    print(f"RDT2-VQ gt_tokens_head: {gt[:10].tolist()}")


def reset_cuda_memory(device):
    torch.cuda.set_device(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)


def print_cuda_memory(name, device):
    allocated = torch.cuda.max_memory_allocated(device) / 1024**3
    reserved = torch.cuda.max_memory_reserved(device) / 1024**3
    free, total = torch.cuda.mem_get_info(device)

    print(f"{name} cuda_peak_allocated_gb: {allocated:.3f}")
    print(f"{name} cuda_peak_reserved_gb: {reserved:.3f}")
    print(f"{name} cuda_current_free_gb: {free / 1024**3:.3f}")
    print(f"{name} cuda_total_gb: {total / 1024**3:.3f}")
