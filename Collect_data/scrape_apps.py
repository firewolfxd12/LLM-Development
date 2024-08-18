import json
import hashlib
from datasets import load_dataset
import os

def save_dataset(dataset, file_path, split="train"):

    # Initialize a list to store the formatted examples
    formatted_data = []
    empty_solutions = 0
    duplicates = 0

    # Set to track unique requests
    unique_requests = set()

    # Iterate through each example in the dataset
    for item in dataset:
        problem_statement = item["question"]
        code_solutions = item["solutions"]

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

            formatted_example = {
                "problem": problem_statement,
                "solution": code_solution
            }

            formatted_data.append(formatted_example)

    # Print the count of processed examples
    print(f"Total {split} examples processed: {len(formatted_data)}\nEmpty solutions: {empty_solutions}\nDuplicates: {duplicates}")

    directory = "Ready_data"
    file_path = os.path.join(directory, file_path)

    # Create the directory if it doesn't exist
    os.makedirs(directory, exist_ok=True)

    # Save the formatted data to a new JSON file
    with open(file_path, mode='w') as writer:
        writer.write(json.dumps(formatted_data, indent=4))
    print(f"Examples saved to {file_path}")


def main():
    # Load both the train and test splits of the APPS dataset
    train_data = load_dataset("codeparrot/apps", "all", split="train")
    valid_data = load_dataset("codeparrot/apps", "all", split="test")
    print(f"Size of the train set: {len(train_data)}. Size of the validation set: {len(valid_data)}")

    save_dataset(train_data, "data_apps.json")

    save_dataset(valid_data, "val_data_apps.json", "validation")

main()