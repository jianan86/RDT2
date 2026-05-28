import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import yaml

import models.rdt_inferencer as rdt_inferencer
from models.rdt_inferencer import RDTInferencer
from shard_inference_utils import (
    NORMALIZER_PATH,
    QWEN_PATH,
    RDT2_FM_PATH,
    RDT2_VQ_PATH,
    load_shard_sample,
    resolve_instruction,
    print_action_report,
    print_cuda_memory,
    reset_cuda_memory,
    use_repo_root,
)


def patch_local_processor():
    original = rdt_inferencer.AutoProcessor.from_pretrained

    def from_local_qwen(*args, **kwargs):
        kwargs.setdefault("padding_side", "left")
        kwargs.setdefault("use_fast", True)
        kwargs.setdefault("local_files_only", True)
        return original(str(QWEN_PATH), **kwargs)

    rdt_inferencer.AutoProcessor.from_pretrained = from_local_qwen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-id", type=int, default=77016)
    parser.add_argument("--instruction", default=None)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    use_repo_root()
    patch_local_processor()
    sample = load_shard_sample(args.sample_id)
    instruction = resolve_instruction(sample, args.instruction)
    print(f"instruction: {instruction}")
    device = torch.device(args.device)
    reset_cuda_memory(device)

    with open("configs/rdt/post_train.yaml", "r") as f:
        model_config = yaml.safe_load(f)

    model = RDTInferencer(
        config=model_config,
        pretrained_path=str(RDT2_FM_PATH),
        normalizer_path=str(NORMALIZER_PATH),
        pretrained_vision_language_model_name_or_path=str(RDT2_VQ_PATH),
        device=args.device,
        dtype=torch.bfloat16,
    )

    result = model.step(
        observations={
            "images": {
                "left_stereo": sample["left_image"],
                "right_stereo": sample["right_image"],
            },
            "state": np.zeros(model_config["common"]["state_dim"]).astype(np.float32),
        },
        instruction=instruction,
    )

    action_chunk = result.detach().cpu()
    print_action_report("RDT2-FM", action_chunk, sample["action"])
    print_cuda_memory("RDT2-FM", device)


if __name__ == "__main__":
    main()
