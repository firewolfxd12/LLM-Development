import os
import json
from accelerate import Accelerator
from datasets import load_dataset, concatenate_datasets
from peft import LoraConfig
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, logging, set_seed
from trl import SFTTrainer
from trl.trainer import ConstantLengthDataset


"""
Fine-Tune Llama-3.1 8B with predefined parameters
"""
model_name = "meta-llama/Meta-Llama-3.1-8B"

def chars_token_ratio(dataset, tokenizer, nb_examples=400):
    """
    Estimate the average number of characters per token in the dataset.
    """
    total_characters, total_tokens = 0, 0
    for _, example in tqdm(zip(range(nb_examples), iter(dataset)), total=nb_examples):
        text = prepare_sample_text(example)
        total_characters += len(text)
        if tokenizer.is_fast:
            total_tokens += len(tokenizer(text).tokens())
        else:
            total_tokens += len(tokenizer.tokenize(text))

    return total_characters / total_tokens


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


def prepare_sample_text(example):
    """Prepare the text from a sample of the dataset."""
    text = f"Question: {example['question']}\n\nAnswer: {json.loads(example['solutions'])[0] if example['solutions'] else ''}"
    return text


def create_datasets(tokenizer):
    apps_dataset_train = load_dataset("codeparrot/apps", "all", split="train")
    apps_dataset_test = load_dataset("codeparrot/apps", "all", split="test")
    dataset = concatenate_datasets([apps_dataset_train, apps_dataset_test])

    dataset = dataset.train_test_split(test_size=0.005, seed=42)
    train_data = dataset["train"]
    valid_data = dataset["test"]
    print(f"Size of the train set: {len(train_data)}. Size of the validation set: {len(valid_data)}")

    chars_per_token = chars_token_ratio(train_data, tokenizer)
    print(f"The character to token ratio of the dataset is: {chars_per_token:.2f}")

    train_dataset = ConstantLengthDataset(
        tokenizer,
        train_data,
        formatting_func=prepare_sample_text,
        infinite=True,
        seq_length=1024,
        chars_per_token=chars_per_token,
    )
    valid_dataset = ConstantLengthDataset(
        tokenizer,
        valid_data,
        formatting_func=prepare_sample_text,
        infinite=False,
        seq_length=1024,
        chars_per_token=chars_per_token,
    )
    return train_dataset, valid_dataset


def run_training(train_data, val_data):
    print("Loading the model")

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    train_data.start_iteration = 0

    print("Starting main loop")

    training_args = TrainingArguments(
        output_dir="./finetuned_llama",
        dataloader_drop_last=True,
        eval_strategy="steps",
        # max_steps=100000,
        eval_steps=500,
        save_steps=500,
        logging_steps=25,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        learning_rate=8e-5,
        lr_scheduler_type="cosine",
        warmup_steps=0,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        max_grad_norm=0.8,
        num_train_epochs=10,
        fp16=False,
        bf16=True,
        weight_decay=0.01,
        run_name="hf_llama_coding_SFT",
        report_to="wandb",
        ddp_find_unused_parameters=False,

        save_total_limit=2,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        load_in_8bit=True,
        device_map={"": Accelerator().process_index}
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        peft_config=lora_config,
        packing=True,
    )

    print_trainable_parameters(trainer.model)

    print("Training...")
    trainer.train()

    print("Saving last checkpoint of the model")
    trainer.model.save_pretrained(os.path.join(training_args.output_dir, "final_checkpoint/"))


def main():
    set_seed(42)
    os.makedirs("./finetuned_llama", exist_ok=True)

    logging.set_verbosity_error()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_dataset, eval_dataset = create_datasets(tokenizer)
    run_training(train_dataset, eval_dataset)


if __name__ == "__main__":
    main()
