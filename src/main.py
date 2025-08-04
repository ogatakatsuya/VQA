from functools import partial

import torch
from torch.optim import AdamW
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
)
from tqdm import tqdm
from transformers.utils.quantization_config import BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType

from dataloader import VQADataset, collate_fn

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-3B-Instruct",
    torch_dtype="auto",
    device_map="auto",
    quantization_config=bnb_config,
)

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

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
optimizer = AdamW(model.parameters(), lr=5e-5)

num_epochs = 1
total_loss = 0
model.train()
for epoch in range(num_epochs):
    for inputs in tqdm(train_loader):
        inputs = inputs.to("cuda")
        labels = inputs.input_ids.clone()
        outputs = model(**inputs, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()
    print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader)}")

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

    print("=" * 50)
    print("VQA Model Results:")
    print("=" * 50)

    for i, (text, input_ids) in enumerate(zip(output_text, inputs.input_ids)):
        print(f"\nSample {i + 1}:")
        print(f"Generated Answer: {text}")
        print("-" * 30)

    print(f"\nProcessed {len(output_text)} samples in this batch.")
    break
