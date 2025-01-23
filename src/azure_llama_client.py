import os
import requests
import json
from typing import List, Optional, Dict, Any
from dotenv import load_dotenv

class LlamaClient:
    def __init__(self, endpoint: str, api_key: str):
        """
        Initialize the Llama client with endpoint and API key.
        
        Args:
            endpoint (str): The Azure endpoint URL for the deployed model
            api_key (str): The API key for authentication
        """
        self.endpoint = endpoint.rstrip('/')
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }

    def generate_text(
        self,
        prompt: str,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        top_p: float = 0.95,
        frequency_penalty: float = 0,
        presence_penalty: float = 0,
        stop_sequences: Optional[List[str]] = None
    ) -> Dict[Any, Any]:
        """
        Generate text using the Llama model.
        
        Args:
            prompt (str): The input prompt for the model
            max_tokens (int): Maximum number of tokens to generate
            temperature (float): Controls randomness (0-1)
            top_p (float): Controls diversity via nucleus sampling (0-1)
            frequency_penalty (float): Penalty for token frequency
            presence_penalty (float): Penalty for token presence
            stop_sequences (List[str], optional): Sequences where generation should stop
            
        Returns:
            Dict: The model's response
        """
        payload = {
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
        }
        
        if stop_sequences:
            payload["stop"] = stop_sequences

        try:
            response = requests.post(
                f"{self.endpoint}/chat/completions?api-version=2024-02-15-preview",
                headers=self.headers,
                json=payload
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error making request: {e}")
            raise

    def generate_chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 1000,
        temperature: float = 0.7,
        top_p: float = 0.95,
        frequency_penalty: float = 0,
        presence_penalty: float = 0,
    ) -> Dict[Any, Any]:
        """
        Generate a chat response using the Llama model.
        
        Args:
            messages (List[Dict[str, str]]): List of message dictionaries with 'role' and 'content'
            max_tokens (int): Maximum number of tokens to generate
            temperature (float): Controls randomness (0-1)
            top_p (float): Controls diversity via nucleus sampling (0-1)
            frequency_penalty (float): Penalty for token frequency
            presence_penalty (float): Penalty for token presence
            
        Returns:
            Dict: The model's response
        """
        payload = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
        }

        try:
            response = requests.post(
                f"{self.endpoint}/chat/completions?api-version=2024-02-15-preview",
                headers=self.headers,
                json=payload
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error making request: {e}")
            raise

# Example usage
if __name__ == "__main__":
    # Load environment variables
    load_dotenv()
    
    api_key = os.getenv("AZURE_API_KEY")
    endpoint = os.getenv("AZURE_ENDPOINT", "https://Llama-3-3-70B-Instruct-TALT.swedencentral.models.ai.azure.com")
    
    if not api_key:
        raise ValueError("AZURE_API_KEY environment variable not set")
    
    # Create client
    client = LlamaClient(endpoint=endpoint, api_key=api_key)
    
    # Example text generation
    try:
        response = client.generate_text(
            prompt="What are three interesting facts about Sweden?",
            max_tokens=500,
            temperature=0.7
        )
        print("\nText Generation Response:")
        print(json.dumps(response, indent=2))
    except Exception as e:
        print(f"Error during text generation: {e}")
    
    # Example chat
    try:
        chat_messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What are three interesting facts about Sweden?"}
        ]
        response = client.generate_chat(
            messages=chat_messages,
            max_tokens=500,
            temperature=0.7
        )
        print("\nChat Response:")
        print(json.dumps(response, indent=2))
    except Exception as e:
        print(f"Error during chat: {e}")
