from openai import OpenAI

client = OpenAI(api_key='your-api-key')


def create_batches(batch_amount=7):
    # Upload the JSONL Files and Create Batches
    for file_number in range(1, batch_amount + 1):
        jsonl_filename = f"Ready_data/Batches/raw_data_apps_{file_number}.jsonl"
        
        batch_input_file = client.files.create(
            file=open(jsonl_filename, "rb"),
            purpose="batch"
        )
        batch_input_file_id = batch_input_file.id  # Access the ID directly
        print(f"File {jsonl_filename} uploaded with ID: {batch_input_file_id}")

        # Create the Batch
        batch = client.batches.create(
            input_file_id=batch_input_file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={
                "description": f"Code solution explanation batch {file_number}"
            }
        )

        batch_id = batch.id  # Access the ID directly
        print(f"Batch created with ID: {batch_id}")


def print_batches():
    # Retrieve and print all batches
    batches = client.batches.list(limit=100)  # Adjust the limit as needed
    for batch in batches.data:
        print(f"Batch ID: {batch.id}")
        print(f"Status: {batch.status}")
        print(f"Description: {batch.metadata.get('description', 'No description')}")
        print(f"Total Requests: {batch.request_counts.total}")
        print(f"Completed Requests: {batch.request_counts.completed}")
        print(f"Failed Requests: {batch.request_counts.failed}")
        print(f"Created At: {batch.created_at}")
        print(f"Completed At: {batch.completed_at}")
        print(f"Error File ID: {batch.error_file_id}")
        print(f"Output File ID: {batch.output_file_id}")
        print("-" * 40)

# Example usage
# create_batches(batch_amount=7)
print_batches()
