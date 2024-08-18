import argparse
import json
import subprocess
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

class SelfDebuggingEnvironment:
    def __init__(self, json_input, max_attempts):
        self.question = json_input['question']
        self.input_output_tests = json_input['input_output']
        self.max_attempts = max_attempts
        self.attempts = 0
        self.model = self.load_model("gpt2")  # Replace with the actual model as needed

    def load_model(self, model_name):
        print(f"Loading model {model_name} from Hugging Face...")
        model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0)
        return generator

    def generate_code(self):
        prompt = f"Question: {self.question}\n\nAnswer: "
        print(f"Generating code from model...")
        response = self.model(prompt, max_new_tokens=512, num_return_sequences=1)
        generated_code = response[0]["generated_text"][len(prompt):]
        return generated_code

    def write_code_to_file(self, code):
        # Write the generated code to a Python file (this will be executed in Docker)
        with open("program.py", "w") as f:
            f.write(code)

    def run_code_in_docker(self, input_data):
        # Run the Docker container with the code and provide the input through stdin
        result = subprocess.run([
            "docker", "run", "--rm", "--network", "none",
            "-v", f"{subprocess.os.getcwd()}:/usr/src/app",  # Mount the current directory
            "safe_execution_env", "python", "/usr/src/app/program.py"
        ], input=input_data, text=True, capture_output=True)

        return result

    def run_tests(self, code):
        # Write the generated code to file
        self.write_code_to_file(code)

        for input_test, expected_output in zip(self.input_output_tests['inputs'], self.input_output_tests['outputs']):
            # Run the code inside the Docker container with the given input test case
            result = self.run_code_in_docker(input_test)

            # Check for any errors in stderr
            if result.stderr.strip():
                return False, f"{result.stderr.strip()}"

            # Check the output from the Docker container and compare with the expected output
            if result.stdout.strip() == expected_output.strip():
                continue  # Test passed, move to the next one
            else:
                return False, f"Test failed: input '{input_test}' produced '{result.stdout.strip()}', expected '{expected_output.strip()}'"

        return True, "All tests passed"

    def run(self):
        while self.attempts < self.max_attempts:
            self.attempts += 1
            generated_code = self.generate_code()

            print(f"\nAttempt {self.attempts} generated code:\n{generated_code}")

            # Syntax Check and Unit Tests
            success, message = self.run_tests(generated_code)
            if success:
                print(f"\nSuccess! The problem is solved after {self.attempts} attempts.")
                print(f"Final Code:\n{generated_code}")
                return generated_code
            else:
                print(f"Error encountered: {message}")
                # If there's an error, re-prompt the model with the error and generated code
                self.question = f"{self.question}\n\n{generated_code}\n\nThe following error occurred:\n{message}\nFix this error and regenerate the code."

        print(f"\nMaximum attempts reached ({self.max_attempts}). The problem was not solved.")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Self-debugging environment to generate and test code using an LLM.")
    parser.add_argument("--json_input", type=str, required=True, help="Path to the JSON input file containing the problem.")
    parser.add_argument("--max_attempts", type=int, required=True, help="Maximum number of attempts for debugging.")
    args = parser.parse_args()

    # Load the JSON input from the file
    with open(args.json_input, 'r') as file:
        json_input = json.load(file)

    # Initialize the debugging environment and start the process
    env = SelfDebuggingEnvironment(json_input, args.max_attempts)
    env.run()
