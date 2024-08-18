# git commit -m feat: "load dataset of humaneval prompts, configure the OpenAI API to test prompts"
from openai import OpenAI
import os
import ast
from datasets import load_dataset
from dotenv import load_dotenv, find_dotenv
from typing import List, Tuple, Dict, Any

# Load datasets
def load_selected_datasets(*dataset_names):
    loaded_datasets = {}

    for dataset_name in dataset_names:
        if dataset_name == "human_eval":
            loaded_datasets["human_eval"] = load_dataset("openai/openai_humaneval", split='test')
        elif dataset_name == "human_eval_plus":
            loaded_datasets["human_eval_plus"] = load_dataset("evalplus/humanevalplus", split='test')
        else:
            print(f"Warning: Dataset '{dataset_name}' is not recognized and will be skipped.")

    return loaded_datasets

# Load ChatGPT API
_ = load_dotenv(find_dotenv())
client = OpenAI(
    api_key=os.environ.get('OPENAI_API_KEY'))
model="gpt-4-turbo-preview"
temperature = 0.3
max_tokens = 150
def evaluate_human_eval(chat_message, exec_globals, idx, dataset):
    entry_point = dataset[idx]['entry_point']

    # define generated code
    try:
        exec(chat_message, exec_globals)
    except Exception as e:
        print(f"Error executing generated function for prompt {idx}: {e}")
        return False

    # run tests for problem
    try:
        exec(dataset[idx]['test'], exec_globals)
    except Exception as e:
        print(f"Error executing test code for prompt {idx}: {e}")
        return False

    # validation
    try:
        exec_globals['check'](exec_globals[entry_point])
    except AssertionError as e:
        print(f"AssertionError in test case for prompt {idx}: {e}")
        return False
    except Exception as e:
        print(f"Error running check function for prompt {idx}: {e}")
        return False
    else:
        print(f"Prompt {idx} passed all tests.")

    return True


# ChatGPT's response to prompt
def ChatGPT(datasets, start=0, end=1):
    failed = []

    for dataset_name, dataset in datasets.items():
        passed = 0
        print('Benchmark: ', dataset_name)
        for idx in range(start, end):
            content = dataset[idx]['prompt']


            # Set up chat completion
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "Only respond with the python function. No other text."},
                    {"role": "user", "content": content}
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )

            chat_message = response.choices[0].message.content

            # Prepare the environment for executing the generated code
            exec_globals = {
                'List': List,
                'Tuple': Tuple,
                'Dict': Dict,
                'Any': Any,
            }

            # Evaluate accuracy based on selected benchmark
            if evaluate_human_eval(chat_message[10:-3], exec_globals, idx, dataset):
                passed = passed + 1
            else:
                failed.append((dataset_name, idx))


        print(f'{dataset_name} accuracy: {passed} / {len(failed) + passed}')

datasets = load_selected_datasets('human_eval')
ChatGPT(datasets, 0, 5)
