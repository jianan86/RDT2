import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from models.normalizer import LinearNormalizer
from shard_inference_utils import (
    NORMALIZER_PATH,
    QWEN_PATH,
    RDT2_VQ_PATH,
    resolve_instruction,
    VQVAE_PATH,
    load_shard_sample,
    print_action_report,
    print_cuda_memory,
    print_token_report,
    reset_cuda_memory,
    use_repo_root,
)
from utils import batch_predict_action
from vqvae.models.multivqvae import MultiVQVAE


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-id", type=int, default=77016)
    parser.add_argument("--instruction", default=None)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    use_repo_root()
    sample = load_shard_sample(args.sample_id)
    instruction = resolve_instruction(sample, args.instruction)
    print(f"instruction: {instruction}")
    device = torch.device(args.device)
    reset_cuda_memory(device)

    processor = AutoProcessor.from_pretrained(
        str(RDT2_VQ_PATH), padding_side="left", use_fast=True, local_files_only=True
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(RDT2_VQ_PATH),
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map=args.device,
        local_files_only=True,
    ).eval()
    vae = MultiVQVAE.from_pretrained(str(VQVAE_PATH), local_files_only=True).eval()
    vae = vae.to(device=device, dtype=torch.float32)
    normalizer = LinearNormalizer.load(str(NORMALIZER_PATH))

    valid_action_id_length = vae.pos_id_len + vae.rot_id_len + vae.grip_id_len
    example = {
        "obs": {
            "camera0_rgb": torch.from_numpy(sample["right_image"]).unsqueeze(0),
            "camera1_rgb": torch.from_numpy(sample["left_image"]).unsqueeze(0),
        },
        "meta": {"num_camera": 2},
    }

    result = batch_predict_action(
        model,
        processor,
        vae,
        normalizer,
        examples=[example],
        valid_action_id_length=valid_action_id_length,
        apply_jpeg_compression=True,
        instruction=instruction,
    )

    action_chunk = result["action_pred"][0].detach().cpu()
    action_tokens = processor.tokenizer.vocab_size - (result["action_ids"][0] + 1)
    action_tokens = torch.clamp(action_tokens, min=0, max=vae.num_embeddings - 1)

    print_action_report("RDT2-VQ", action_chunk, sample["action"])
    print_token_report(action_tokens, sample["action_token"])

    gt_tokens = torch.from_numpy(sample["action_token"]).to(device=device, dtype=torch.long).unsqueeze(0)
    gt_recon = normalizer["action"].unnormalize(vae.decode(gt_tokens))[0].detach().cpu()
    print_action_report("RDT2-VQ gt_token_decode", gt_recon, sample["action"])
    print_cuda_memory("RDT2-VQ", device)


if __name__ == "__main__":
    main()
