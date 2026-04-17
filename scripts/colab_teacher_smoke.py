from __future__ import annotations

import argparse
from pathlib import Path

import torch
import transformers
from transformers import AutoConfig, AutoProcessor, Qwen3VLForConditionalGeneration, Qwen3VLMoeForConditionalGeneration


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        type=Path,
        required=True,
        help="Local directory containing the downloaded Qwen3-VL checkpoint.",
    )
    parser.add_argument(
        "--image-url",
        type=str,
        default="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
        help="Remote image URL for the smoke test.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Describe this image in one sentence.",
        help="Prompt text for the smoke test.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"[smoke] torch={torch.__version__}")
    print(f"[smoke] cuda_available={torch.cuda.is_available()}")
    print(f"[smoke] transformers={transformers.__version__}")
    config = AutoConfig.from_pretrained(args.model_dir.as_posix())
    print(f"[smoke] config_model_type={getattr(config, 'model_type', 'unknown')}")
    architectures = set(config.architectures or [])
    print(f"[smoke] architectures={sorted(architectures)}")
    if getattr(config, "quantization_config", None) is not None:
        raise SystemExit(
            "[smoke] Detected a quantized checkpoint in this model directory. "
            "For the 98GB Colab GPU path we want the full-precision "
            "Qwen/Qwen3-VL-30B-A3B-Instruct checkpoint, not an AWQ/GPTQ variant. "
            "Please clear the wrong cached folder and rerun setup."
        )
    model_cls = (
        Qwen3VLMoeForConditionalGeneration
        if "Qwen3VLMoeForConditionalGeneration" in architectures or getattr(config, "model_type", None) == "qwen3_vl_moe"
        else Qwen3VLForConditionalGeneration
    )
    if getattr(config, "model_type", None) not in {"qwen3_vl", "qwen3_vl_moe"}:
        raise SystemExit(
            "[smoke] This model directory does not contain a Qwen3-VL checkpoint. "
            f"Found model_type={getattr(config, 'model_type', 'unknown')} instead. "
            "Most likely Colab is pointing at an older cached model directory."
        )
    print(f"[smoke] model_class={model_cls.__name__}")
    model = model_cls.from_pretrained(
        args.model_dir.as_posix(),
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        attn_implementation="sdpa",
    )
    processor = AutoProcessor.from_pretrained(args.model_dir.as_posix())

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": args.image_url},
                {"type": "text", "text": args.prompt},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs.pop("token_type_ids", None)
    if torch.cuda.is_available():
        inputs = {key: value.to("cuda") if hasattr(value, "to") else value for key, value in inputs.items()}
    generated_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    print(output_text[0])


if __name__ == "__main__":
    main()
