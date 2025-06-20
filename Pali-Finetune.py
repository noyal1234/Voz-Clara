from datasets import load_dataset
import torch
from transformers import PaliGemmaProcessor, PaliGemmaForConditionalGeneration, Trainer, TrainingArguments, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig
import os

# Set fine-tuning options
USE_LORA = False
USE_QLORA = False 
FREEZE_VISION = False 

ds = load_dataset('lmms-lab/VizWiz-VQA', split="train")

model_id = "google/paligemma2-3b-pt-448" 
processor = PaliGemmaProcessor.from_pretrained(model_id)

device = "cuda" if torch.cuda.is_available() else "cpu"

# Get the special token id for "<image>" if needed (it’s used to mark the image location in the text prompt)
image_token = processor.tokenizer.convert_tokens_to_ids("<image>")

def collate_fn(examples):
    # For each example, prepend a task-specific prefix to the question.
    # Here we assume each example has a "question" field and a "multiple_choice_answer" field.
    texts = ["<image>answer en " + example["question"] for example in examples]
    labels = [example["multiple_choice_answer"] for example in examples]
    # Convert the image field to RGB (assumes the "image" field is a PIL Image)
    images = [example["image"].convert("RGB") for example in examples]
    tokens = processor(text=texts, images=images, suffix=labels,
                       return_tensors="pt", padding="longest")
    # Cast tensors to bfloat16 and move to device
    tokens = tokens.to(torch.bfloat16).to(device)
    return tokens

# Load the model. Depending on whether LoRA or QLoRA is used, configure accordingly.
if USE_LORA or USE_QLORA:
    lora_config = LoraConfig(
        r=8,
        target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    if USE_QLORA:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_type=torch.bfloat16
        )
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        model_id, 
        device_map="auto",
        quantization_config=bnb_config if USE_QLORA else None,
        torch_dtype=torch.bfloat16
    )
    model = get_peft_model(model, lora_config)
    model = model.to(device)
    model.print_trainable_parameters()
else:
    model = PaliGemmaForConditionalGeneration.from_pretrained(model_id, device_map="auto").to(device)
    model = model.to(device)
    if FREEZE_VISION:
        for param in model.vision_tower.parameters():
            param.requires_grad = False
        for param in model.multi_modal_projector.parameters():
            param.requires_grad = False


args = TrainingArguments(
    num_train_epochs=3,                  
    remove_unused_columns=False,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    warmup_steps=2,
    learning_rate=2e-5,
    weight_decay=1e-6,
    adam_beta2=0.999,
    logging_steps=100,
    optim="adamw_hf",
    save_strategy="steps",
    save_steps=1000,
    save_total_limit=1,
    push_to_hub=True,
    output_dir="paligemma_vizwiz",
    bf16=True,
    report_to=["tensorboard"],
    dataloader_pin_memory=False
)

trainer = Trainer(
    model=model,
    train_dataset=ds,
    data_collator=collate_fn,
    args=args
)

trainer.train()
