from __future__ import annotations

import argparse
from pathlib import Path

import torch
import transformers
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        type=Path,
        required=True,
        help="Local directory containing the downloaded Qwen2.5-VL checkpoint.",
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
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_dir.as_posix(),
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
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
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda" if torch.cuda.is_available() else "cpu")
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
