import json
import re
import random
from datasets import load_dataset

# Function to remove docstrings from the code
def remove_docstring(code):
    # Pattern to match single-line and multi-line docstrings and the following blank line
    pattern = r' *r?("""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\')[^\n]*\n'
    # Remove the docstrings and the following blank line using regex
    return re.sub(pattern, '', code)

# Load the CodeSearchNet dataset for Python using Hugging Face
dataset = load_dataset("code_search_net", "python", split="train", trust_remote_code=True)

# Initialize a list to store the formatted examples
formatted_data = []

# Iterate through each example in the dataset
for example in dataset:
    code = example.get('func_code_string')
    docstring = example.get('func_documentation_string')
    
    # Ensure both problem (docstring) and solution (code) exist
    if code and docstring:
        cleaned_code = remove_docstring(code)
        formatted_example = {
            "problem": docstring,
            "solution": cleaned_code.strip(),  # Remove leading/trailing whitespace
            "explanation": docstring,
        }
        formatted_data.append(formatted_example)

# Print some examples
random_examples = random.sample(formatted_data, 5)  # Randomly select 5 examples
for i, example in enumerate(random_examples):
    print(f"Random Example {i+1}:")
    print(f"Problem: \n{example['problem']}")
    print(f"Solution: \n{example['solution']}")
    print(f"Explanation: \n{example['explanation']}")
    print("-" * 50)

# Print the count of processed examples
print(f"Total examples processed: {len(formatted_data)}")

# Save the formatted data to a new JSON file
with open("data_code_search_net.json", mode='w') as writer:
    writer.write(json.dumps(formatted_data, indent=4))
