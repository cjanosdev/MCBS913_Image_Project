from datetime import datetime
from PIL import Image

class Conversation:
    def __init__(self):
        self.history_log = []
        self.conversation = []

    def format_response(self, response: str) -> str:
        """ Format the response from DeepThought for display """
        if not response:
            return "⚠️ No response received."
        return f"\n📌 **DeepThought's Response:**\n{response}"

    def log_interaction(self, role: str, content: dict):
        """ Log the conversation interaction for later retrieval or export """
        self.history_log.append({
            "timestamp": datetime.now().isoformat(),
            "role": role,
            "content": content
        })

    def build_conversation(self, user_input_text: str, image: str = None) -> dict:
        """ Build a conversation entry with text and optionally an image """
        convo = {
            "role": "user",
            "content": [{"type": "text", "text": user_input_text}]
        }
        if image:
            convo["content"].append({"type": "image", "image": Image.open(image)})
        return convo
