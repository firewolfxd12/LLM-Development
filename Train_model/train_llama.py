import json
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, TrainerCallback
from trl import SFTTrainer
from trl.trainer import ConstantLengthDataset
from datasets import load_dataset
from peft import LoraConfig
import torch
import os
import wandb
from torch.amp import autocast
import torch.backends.cudnn as cudnn

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

max_seq_length = 512

def format_data(entry):
    task = entry.get("problem", "").strip()
    code = entry.get("solution", "").strip()
    explanation = entry.get("explanation", "").strip()

    formatted_entry = f"<s><task> {task} </task>\n<code> {code} </code>\n<explanation> {explanation} </explanation>"
    
    return formatted_entry


def load_dataset(tokenizer, val=False):
    if val:
        file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Collect_data', 'Ready_data', 'val_data_code_search_net.json'))
    else:
        file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Collect_data', 'Ready_data', 'data_code_search_net.json'))

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist. Please ensure that you have downloaded the finetuning data")

    # Load and prepare the dataset
    with open(file_path, 'r') as f:
        data = json.load(f)

    data = ConstantLengthDataset(
        tokenizer,
        dataset=data,
        infinite=True,
        seq_length=max_seq_length,
        formatting_func=format_data,
        # chars_per_token=chars_per_token,
    )

    return data


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


def main():
    # Ensure CUDA is available
    if not torch.cuda.is_available():
        raise RuntimeError("No GPU found. A GPU is needed for quantization.")

    # Enable TF32
    torch.backends.cuda.matmul.allow_tf32 = True
    cudnn.allow_tf32 = True

    # Define special tokens
    SPECIAL_TOKENS = {
        'pad_token': '<pad>',
        "additional_special_tokens": ["<s>", "<task>", "</task>", "<code>", "</code>", "<explanation>", "</explanation>"]
    }

    # Use BitsAndBytesConfig for 8-bit quantization
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,  # Enable 8-bit loading
        llm_int8_threshold=6.0  # Optional: Threshold for dynamic quantization, adjust if needed
    )

    # Load model and setup chat format
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Meta-Llama-3.1-8B", 
        low_cpu_mem_usage=True, 
        torch_dtype="bfloat16", 
        quantization_config=quantization_config, 
        device_map="auto"
    ) # load_in_8bit=True, torch_dtype="bfloat16",

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B")
    tokenizer.add_special_tokens(SPECIAL_TOKENS)
    model.resize_token_embeddings(len(tokenizer))

    train_data = load_dataset(tokenizer)
    val_data = load_dataset(tokenizer, val=True)

    print_trainable_parameters(model)

    train_data.start_iteration = 0

    # Configuration for the PEFT
    peft_config = LoraConfig(
        r=16,  # Increasing r to 32 for more expressive power; reduce if running out of memory
        lora_alpha=32,  # A higher alpha value can lead to better adaptation but increases memory usage
        lora_dropout=0.05,  # Slightly higher dropout to regularize and prevent overfitting
        target_modules=["q_proj", "k_proj", "v_proj", "up_proj", "down_proj", "o_proj", "gate_proj"], 
        bias="none",  # Default setting; adjust if needed based on your specific model
        task_type="CAUSAL_LM",  # Assuming you're fine-tuning for a causal language modeling task
    )

    # Define TrainingArguments with careful consideration of resources and task requirements
    training_args = TrainingArguments(
        learning_rate=8e-5,
        lr_scheduler_type="cosine",
        per_device_train_batch_size=1,  # Increase batch size
        gradient_accumulation_steps=2,  # Adjust as needed
        max_grad_norm=2.0,  # Enable gradient clipping
        num_train_epochs=1,  # Increase epochs for better learning
        bf16=True,  # Continue using BF16
        logging_steps=25,  # Reduce frequency of logging
        optim="adamw_torch",
        weight_decay=0.01,
        warmup_steps=500,  # Increase warmup for stable gradients
        output_dir="./finetuned_llama",
        seed=42,
        save_total_limit=2,
        save_steps=500,  # Adjust checkpoint saving frequency
        run_name="llama_coding_SFT",
        report_to="wandb",
        dataloader_num_workers=2,
    )

    class WandbCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs is not None:
                wandb.log(logs)

    # Define which wandb project to log to and name your run
    run = wandb.init(project="llama_coding_SFT", name="llama_3.1_8b_base_debug")

    # Start training with automatic mixed precision
    with autocast(device_type='cuda', dtype=torch.bfloat16):
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_data,
            eval_dataset=val_data,
            max_seq_length=max_seq_length,
            peft_config=peft_config,
            args=training_args,
            callbacks=[WandbCallback()],
            packing=True
        )
        
        print("Training...")
        trainer.train()

        print("Saving last checkpoint of the model")
        trainer.model.save_pretrained(os.path.join("./finetuned_llama", "final_checkpoint/"))

    run.finish()

if __name__ == '__main__':
    main()
