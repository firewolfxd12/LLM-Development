from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import torch

# Path to the model checkpoint
model_checkpoint = "finetuned_llama/checkpoint-5000"

# Load the model and tokenizer
model = AutoModelForCausalLM.from_pretrained(model_checkpoint, torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

# Initialize the pipeline
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0)

# Example input question (in the same format used during training)
input_question = "You have $n$ barrels lined up in a row, numbered from left to right from one. Initially, the $i$-th barrel contains $a_i$ liters of water.\n\nYou can pour water from one barrel to another. In one act of pouring, you can choose two different barrels $x$ and $y$ (the $x$-th barrel shouldn't be empty) and pour any possible amount of water from barrel $x$ to barrel $y$ (possibly, all water). You may assume that barrels have infinite capacity, so you can pour any amount of water in each of them. \n\nCalculate the maximum possible difference between the maximum and the minimum amount of water in the barrels, if you can pour water at most $k$ times.\n\nSome examples:   if you have four barrels, each containing $5$ liters of water, and $k = 1$, you may pour $5$ liters from the second barrel into the fourth, so the amounts of water in the barrels are $[5, 0, 5, 10]$, and the difference between the maximum and the minimum is $10$;  if all barrels are empty, you can't make any operation, so the difference between the maximum and the minimum amount is still $0$. \n\n\n-----Input-----\n\nThe first line contains one integer $t$ ($1 \\le t \\le 1000$)\u00a0\u2014 the number of test cases.\n\nThe first line of each test case contains two integers $n$ and $k$ ($1 \\le k < n \\le 2 \\cdot 10^5$)\u00a0\u2014 the number of barrels and the number of pourings you can make.\n\nThe second line contains $n$ integers $a_1, a_2, \\dots, a_n$ ($0 \\le a_i \\le 10^{9}$), where $a_i$ is the initial amount of water the $i$-th barrel has.\n\nIt's guaranteed that the total sum of $n$ over test cases doesn't exceed $2 \\cdot 10^5$.\n\n\n-----Output-----\n\nFor each test case, print the maximum possible difference between the maximum and the minimum amount of water in the barrels, if you can pour water at most $k$ times.\n\n\n-----Example-----\nInput\n2\n4 1\n5 5 5 5\n3 2\n0 0 0\n\nOutput\n10\n0"

# Create the input prompt following the training format
input_prompt = f"Question: {input_question}\n\nAnswer:"

# Generate sample texts with increased complexity
responses = pipe(input_prompt, max_length=1000, num_return_sequences=1, truncation=True)

# Print the generated text
for i, response in enumerate(responses):
    print(f"Generated Text {i+1}:\n{response['generated_text']}\n")
