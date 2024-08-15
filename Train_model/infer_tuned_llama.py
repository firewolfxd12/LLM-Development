from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, TextStreamer
import torch
import os

# Set the device to use the second GPU (RTX 2060)
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Load the model in 8-bit precision with CPU offloading
model = AutoModelForCausalLM.from_pretrained(
    "./finetuned_llama/checkpoint-19000", 
    device_map="auto",  # Automatically map layers to available devices
    torch_dtype=torch.float16,  # Use half precision
    load_in_4bit=True,  # Load in 8-bit precision
)

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("./finetuned_llama/checkpoint-19000")

# Initialize the pipeline with the loaded model
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=0  # This should be 0 since we're using CUDA_VISIBLE_DEVICES
)

# Initialize a TextStreamer for live text generation output
streamer = TextStreamer(tokenizer)

# Generate sample text with real-time streaming
inputs = "<s><task> Given a string `s`, return the longest palindromic substring in `s`.\n </task>\n<code>"
_ = pipe(inputs, max_length=75, num_return_sequences=3, truncation=True, streamer=streamer)
