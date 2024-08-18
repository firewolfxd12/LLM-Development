# git commit -m feat: "load dataset of humaneval prompts, configure the OpenAI API to test prompts"
from openai import OpenAI
import os
from datasets import load_dataset
from dotenv import load_dotenv, find_dotenv
from typing import List, Tuple, Dict, Any  # Import commonly used type hints

# Load the dataset
dataset = load_dataset("openai/openai_humaneval", split="test")

# Load ChatGPT API
_ = load_dotenv(find_dotenv())
client = OpenAI(
    api_key=os.environ.get('OPENAI_API_KEY'),
)

model = "gpt-4-turbo-preview"
temperature = 0.3
max_tokens = 150


def evaluate(chat_message, exec_globals, idx):
    try:
        exec(chat_message, exec_globals)
    except Exception as e:
        print(f"Error executing generated function for prompt {idx}: {e}")
        return False

    try:
        exec(dataset[idx]['test'], exec_globals)
    except Exception as e:
        print(f"Error executing test code for prompt {idx}: {e}")
        return False

    entry_point = dataset[idx]['entry_point']

    try:
        exec_globals['check'](
            exec_globals[entry_point])
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
def ChatGPTResponse(start=0, end=1):
    passed = 0
    failed = []

    for idx in range(start, end):
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Only respond with the python function. No other text."},
                {"role": "user", "content": dataset[idx]['prompt']}
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

        if evaluate(chat_message[10:-3], exec_globals, idx):
            passed = passed + 1
        else:
            failed.append(idx)

    print(f'Accuracy: {passed} / {len(failed) + passed}')


ChatGPTResponse(0, 10)
