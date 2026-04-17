from __future__ import annotations

import argparse
from pathlib import Path

import torch
import transformers
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


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
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_dir.as_posix(),
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        attn_implementation="sdpa",
    )
    processor = AutoProcessor.from_pretrained(args.model_dir.as_posix())

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "url": args.image_url},
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
