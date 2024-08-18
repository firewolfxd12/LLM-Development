import subprocess
import shutil

def initialize_docker_environment():
    # Check if Docker is installed
    if not shutil.which("docker"):
        print("Docker is not installed. Please install Docker and try again.")
        exit(1)

    # Check if the Docker image 'safe_execution_env' exists
    docker_image_exists = subprocess.run(
        ["docker", "images", "-q", "safe_execution_env"], 
        capture_output=True, 
        text=True
    ).stdout.strip()

    if not docker_image_exists:
        print("Docker image 'safe_execution_env' does not exist. Building the image...")
        try:
            # Build the Docker image
            subprocess.run(["docker", "build", "-t", "safe_execution_env", "."], check=True)
            print("Docker image 'safe_execution_env' built successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Error building Docker image: {e}")
            exit(1)
    else:
        print("Docker image 'safe_execution_env' already exists.")

if __name__ == "__main__":
    initialize_docker_environment()
