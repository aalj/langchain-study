import os
from dotenv import load_dotenv

def load_env(key):
    # Load environment variables from .env file
    load_dotenv()

    # Get environment variables
    api_key = os.getenv(key)

    # Check if environment variables are present
    if not api_key:
        raise ValueError("Environment variables are missing.")

    # Return environment variables as dictionary
    return api_key

if __name__ == '__main__':
    print(load_env('abc'))