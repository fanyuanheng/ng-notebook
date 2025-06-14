import requests
from typing import Dict, List, Any
from ..core.config import API_URL, CHAT_TIMEOUT

class APIClient:
    def __init__(self, base_url: str = API_URL):
        self.base_url = base_url

    def upload_file(self, file_data: bytes, filename: str) -> bool:
        """Upload a file to the backend."""
        files = {"file": (filename, file_data)}
        response = requests.post(f"{self.base_url}/upload", files=files, timeout=CHAT_TIMEOUT)
        return response.status_code == 200

    def send_chat_message(self, message: str, chat_history: List[Dict[str, str]]) -> Dict[str, Any]:
        """Send a chat message to the backend."""
        response = requests.post(
            f"{self.base_url}/chat",
            json={"message": message, "chat_history": chat_history},
            timeout=CHAT_TIMEOUT
        )
        if response.status_code == 200:
            return response.json()
        raise Exception("Failed to get response from chat API") 