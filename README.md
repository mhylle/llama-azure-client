# Llama Azure Client

A Python client for interacting with Azure-deployed Llama 3.3 models.

## Setup

1. Create the conda environment:
```bash
conda env create -f environment.yml
```

2. Activate the environment:
```bash
conda activate llama_azure
```

3. Copy `.env.template` to `.env` and fill in your Azure API key:
```bash
cp .env.template .env
```

## Usage

```python
from src.azure_llama_client import LlamaClient
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Initialize client
client = LlamaClient(
    endpoint="https://Llama-3-3-70B-Instruct-TALT.swedencentral.models.ai.azure.com",
    api_key=os.getenv("AZURE_API_KEY")
)

# Generate text
response = client.generate_text(
    prompt="What are three interesting facts about Sweden?",
    max_tokens=500,
    temperature=0.7
)
print(response)
```

## Features

- Text generation with customizable parameters
- Chat-style interactions with message history
- Error handling and retry logic
- Configurable generation parameters (temperature, top_p, etc.)