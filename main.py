from sympy import use
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
)
import torch
from functools import partial

from dataloader import VQADataset, collate_fn

# default: Load the model on the available device(s)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-3B-Instruct",
    torch_dtype="auto",
    device_map="auto",
)

# プロセッサーを初期化
processor = AutoProcessor.from_pretrained(
    "Qwen/Qwen2.5-VL-3B-Instruct",
    use_fast=True,
    padding_side="left",
)

train_dataset = VQADataset(
    df_path="./data/train.json",
    image_dir="./data/train",
    has_answer=True,
)
test_dataset = VQADataset(
    df_path="./data/valid.json",
    image_dir="./data/valid",
    has_answer=False,
)

collate_with_processor = partial(collate_fn, processor=processor)

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True,
    collate_fn=collate_with_processor,
    num_workers=0,
    pin_memory=True,
)
test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=8,
    shuffle=False,
    collate_fn=collate_with_processor,
    num_workers=0,
    pin_memory=False,
)

model.eval()
for inputs in test_loader:
    inputs = inputs.to("cuda")

    generated_ids = model.generate(**inputs, max_new_tokens=128)

    generated_ids_trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    # 結果を出力
    print("=" * 50)
    print("VQA Model Results:")
    print("=" * 50)

    for i, (text, input_ids) in enumerate(zip(output_text, inputs.input_ids)):
        print(f"\nSample {i + 1}:")
        print(f"Generated Answer: {text}")
        print("-" * 30)

    print(f"\nProcessed {len(output_text)} samples in this batch.")
    break
