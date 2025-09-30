import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

dataset = load_dataset("daekeun-ml/naver-news-summarization-ko")

BASE_MODEL = "google/gemma-2b-it"

# 토크나이저
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.padding_side = 'right'
tokenizer.pad_token = tokenizer.eos_token

def formatting_func(example):
    text = f"""<bos><start_of_turn>user
다음 글을 요약해주세요:

{example['document']}<end_of_turn>
<start_of_turn>model
{example['summary']}<end_of_turn><eos>"""
    return text

# LoRA 설정
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "o_proj", "k_proj", "v_proj", 
                    "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM",
)

# 모델 로드
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
)

model.gradient_checkpointing_enable()
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# 학습 설정
training_args = TrainingArguments(
    output_dir="outputs",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    max_steps=50,
    num_train_epochs=1,
    learning_rate=2e-4,
    bf16=True,
    logging_steps=10,
    save_strategy="steps",
    save_steps=50,
    push_to_hub=False,
    report_to='none',
    gradient_checkpointing=True,
    # resume_from_checkpoint="outputs/checkpoint-347",
)

# SFTTrainer - 최소한의 파라미터만 사용
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset['train'],
    formatting_func=formatting_func,
)

# 토크나이저 수동 설정
trainer.tokenizer = tokenizer

trainer.train()

# yaml에서 설정
# from accelerate import Accelerator
# # 학습 완료 후 저장
# accelerator = Accelerator()
# accelerator.wait_for_everyone()

# if accelerator.is_main_process:
#     print("Saving LoRA adapter...")
#     unwrapped_model = accelerator.unwrap_model(trainer.model)
#     unwrapped_model.save_pretrained("lora_adapter_final")
#     tokenizer.save_pretrained("lora_adapter_final")
#     print("Done!")