import openai
import asyncio
import aiohttp
from tqdm import tqdm
import json
from datasets import load_dataset
import os
import aiofiles

# Set your OpenAI API key
openai.api_key = 'your-openai-api-key'  # Replace with your actual API key

# Function to generate explanation using GPT-4-turbo asynchronously with retries
async def generate_explanation(session, problem_statement, code_solution, retries=3, delay=5):
    prompt = (
        f"Here is a programming problem and its solution:\n\n"
        f"Problem:\n{problem_statement}\n\n"
        f"Solution:\n{code_solution}\n\n"
        f"Please provide a short summary of the code solution:"
    )

    for attempt in range(retries):
        try:
            response = await session.post(
                'https://api.openai.com/v1/chat/completions',
                headers={
                    'Authorization': f'Bearer {openai.api_key}',
                    'Content-Type': 'application/json',
                },
                json={
                    'model': 'gpt-4-turbo',
                    'messages': [
                        {"role": "system", "content": "You are a helpful assistant who explains code in a very short way."},
                        {"role": "user", "content": prompt}
                    ],
                    'max_tokens': 300,
                    'temperature': 0.7,
                },
                timeout=30  # Timeout in seconds
            )

            # Ensure the response was successful
            if response.status != 200:
                print(f"Request failed with status code {response.status}")
                print(await response.text())  # Print the response text for debugging
                await asyncio.sleep(delay)  # Wait before retrying
                continue

            response_json = await response.json()

            # Check if 'choices' is in the response
            if 'choices' not in response_json:
                print(f"Unexpected response format: {response_json}")
                return None

            explanation = response_json['choices'][0]['message']['content'].strip()
            return explanation

        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            print(f"An error occurred: {e}. Retrying in {delay} seconds...")
            await asyncio.sleep(delay)  # Wait before retrying

    print("Max retries exceeded. Skipping this request.")
    return None

# Function to process the dataset and generate explanations in JSON format asynchronously
async def process_dataset(dataset_split, output_json_path, rate_limit_per_second, save_interval=10):
    semaphore = asyncio.Semaphore(rate_limit_per_second)  # Limit concurrent requests per second

    # Start the JSON array
    async with aiofiles.open(output_json_path, 'w') as output_file:
        await output_file.write('[')

    async with aiohttp.ClientSession() as session:
        for index, item in enumerate(tqdm(dataset_split, desc="Generating Explanations")):
            problem_statement = item['question']
            code_solutions = json.loads(item['solutions'])

            for solution in code_solutions:

                await semaphore.acquire()  # Acquire semaphore before making the request

                explanation = await generate_explanation(session, problem_statement, solution)
                semaphore.release()

                if explanation:
                    result = {
                        "problem": problem_statement,
                        "solution": solution,
                        "explanation": explanation
                    }

                    # Append result to the file
                    async with aiofiles.open(output_json_path, 'a') as output_file:
                        # Add a comma if it's not the first result
                        if index > 0:
                            await output_file.write(',')
                        await output_file.write(json.dumps(result, indent=4))

    # Close the JSON array
    async with aiofiles.open(output_json_path, 'a') as output_file:
        await output_file.write(']')

    print(f"Final explanations saved to {output_json_path}")

# Main function to execute the script
async def main():
    # Load the APPS dataset from Hugging Face
    train_val_split = load_dataset('codeparrot/apps', 'all', split='train')
    val_test_split = load_dataset('codeparrot/apps', 'all', split='test')

    # Set your rate limit (e.g., 60 requests per minute -> 1 request per second)
    rate_limit_per_second = 1  # Adjust according to your API limits

    # Create the directory if it doesn't exist
    directory = "Ready_data"
    os.makedirs(directory, exist_ok=True)

    # Process and save test set
    file_path = os.path.join(directory, 'data_apps.json')
    await process_dataset(train_val_split, file_path, rate_limit_per_second)

    # Process and save validation set
    file_path = os.path.join(directory, 'val_data_apps.json')
    await process_dataset(val_test_split, file_path, rate_limit_per_second)

# Run the script
asyncio.run(main())
