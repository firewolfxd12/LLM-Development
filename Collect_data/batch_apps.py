import json
import os
import hashlib
from openai import OpenAI
from datasets import load_dataset, concatenate_datasets
import tiktoken

max_total_tokens = 8192 
max_output_tokens = 750
max_input_tokens = max_total_tokens - max_output_tokens  # Tokens available for the input (problem + solution)
truncated = 0

tokenizer = tiktoken.get_encoding("cl100k_base")
def truncate_prompt(prompt, max_tokens=max_input_tokens):
    global truncated
    tokens = tokenizer.encode(prompt)
    if len(tokens) > max_tokens:
        truncated += 1
        truncated_tokens = tokens[max_tokens:]
        truncated_statement = tokenizer.decode(truncated_tokens)
        return truncated_statement
    return prompt

# Function to get the size of the current file
def get_file_size(filename):
    return os.path.getsize(filename)

# Step 1: Prepare the JSONL Files

# Load both the train and test splits of the APPS dataset
apps_dataset_train = load_dataset("codeparrot/apps", "all", split="train")
apps_dataset_test = load_dataset("codeparrot/apps", "all", split="test")

# Combine the datasets into one list for easier processing
combined_dataset = concatenate_datasets([apps_dataset_train, apps_dataset_test])

# Create the directories if it doesn't exist
directory = "Ready_data"
os.makedirs(directory, exist_ok=True)
directory = "Ready_data/Batches"
os.makedirs(directory, exist_ok=True)

# Define a counter for unique requests and file handling
request_counter = 0
counter_per_batch = 0
empty_solutions = 0
file_counter = 1
duplicates = 0
jsonl_filename = f"Ready_data/Batches/raw_data_apps_{file_counter}.jsonl"

# Set to track unique requests
unique_requests = set()

# Create the first JSONL file
f = open(jsonl_filename, "w")

for i in range(len(combined_dataset)):
    problem_statement = combined_dataset[i]["question"]
    code_solutions = combined_dataset[i]["solutions"]

    # Check if code_solutions is None or empty
    if code_solutions:
        code_solutions = json.loads(code_solutions)
    else:
        empty_solutions += 1
        continue
    
    # Iterate over each solution
    for code_solution in code_solutions:
        # Create a unique identifier for the request
        request_hash = hashlib.sha256((problem_statement + code_solution).encode('utf-8')).hexdigest()

        # Check if this request is a duplicate
        if request_hash in unique_requests:
            duplicates += 1
            continue

        # Add the request to the set of unique requests
        unique_requests.add(request_hash)   

        request_counter += 1

        prompt = (
            f"Here is a programming problem and its solution:\n\n"
            f"Problem:\n{problem_statement}\n\n"
            f"Solution:\n{code_solution}\n\n"
            f"Please provide a short summary of the code solution:"
        )

        prompt = truncate_prompt(prompt)

        # Create the JSON object for each request
        request_data = {
            "custom_id": f"request-{request_counter}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "gpt-4o",  # Choose the model you want to use
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant who explains Python code."},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": max_output_tokens,  # Adjust as needed
                "temperature": 0.2  # Adjust as needed
            }
        }

        # Serialize the JSON object to a string to calculate its size
        request_json_str = json.dumps(request_data)
        request_size = len(request_json_str.encode('utf-8'))

        # Check if we need to split the file (based on size or number of requests)
        if request_counter % 50000 == 0 or (get_file_size(jsonl_filename) + request_size) > 100 * 1024 * 1024:
            f.close()
            print(f"{jsonl_filename} created with {counter_per_batch} requests.")
            counter_per_batch = 0
            file_counter += 1
            jsonl_filename = f"Ready_data/Batches/raw_data_apps_{file_counter}.jsonl"
            f = open(jsonl_filename, "w")

        # Write the JSON object to the JSONL file
        counter_per_batch += 1
        f.write(request_json_str + "\n")

# Close the last file
f.close()
print(f"{jsonl_filename} created with {counter_per_batch} requests.")
print(f"Unique requests: {request_counter}\nEmpty solutions: {empty_solutions}\nDuplicates: {duplicates}\nTruncated prompts: {truncated}")

client = OpenAI(api_key='your-openai-api-key')

# Step 2: Upload the JSONL Files and Create Batches
for file_number in range(1, file_counter + 1):
    jsonl_filename = f"Ready_data/Batches/raw_data_apps_{file_number}.jsonl"
    
    batch_input_file = client.files.create(
        file=open(jsonl_filename, "rb"),
        purpose="batch"
    )
    batch_input_file_id = batch_input_file['id']
    print(f"File {jsonl_filename} uploaded with ID: {batch_input_file_id}")

    # Step 3: Create the Batch
    batch = client.batches.create(
        input_file_id=batch_input_file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={
            "description": f"Code solution explanation batch {file_number}"
        }
    )

    batch_id = batch['id']
    print(f"Batch created with ID: {batch_id}")