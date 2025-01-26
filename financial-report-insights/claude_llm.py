
import requests

class ClaudeSonetLLM:
    def __init__(self, api_key: str, model: str = "claude-sonet-3.5"):
        self.api_key = api_key
        self.model = model
        self.url = "https://api.anthropic.com/v1/claude"  # Claude Sonet API URL

    def generate(self, prompt: str) -> str:
        # Request payload to send to Claude API
        payload = {
            "model": self.model,
            "prompt": prompt,
            "max_tokens": 150  # Adjust as needed
        }

        # Headers with the API key for authentication
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        # Send the request to Claude's API
        response = requests.post(self.url, json=payload, headers=headers)

        if response.status_code == 200:
            result = response.json()
            return result.get("text", "")  # Return the text response from Claude
        else:
            raise Exception(f"Error: {response.status_code}, {response.text}")
