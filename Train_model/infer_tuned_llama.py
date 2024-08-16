from transformers import pipeline
import torch

# Initialize the pipeline with a model checkpoint
pipe = pipeline("text-generation", model="finetuned_llama\checkpoint-26500", device=0, torch_dtype=torch.bfloat16)

# Generate sample texts with increased complexity
responses = pipe("Infer the DataType from obj", max_length=300, num_return_sequences=3, truncation=True)

# Print the generated text
for i, response in enumerate(responses):
    print(f"Generated Text {i+1}:\n{response['generated_text']}\n")
