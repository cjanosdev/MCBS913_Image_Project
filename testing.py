from deepthought import *
from PIL import Image
from tqdm import tqdm
import os
import logging
import json
import textwrap
from datetime import datetime

logging.basicConfig(level=logging.INFO)

AUTH_KEY = "7ceac845-58cb-4260-a558-e83fcee7d776"
MODEL_LLAMA_VIS = "Llama-3.2-11B-Vision-Instruct"
MODEL_LLAMA = "Meta-Llama-3.1-8B-Instruct"

conversation = []
history_log = []

# -------- Conversation Helpers --------

def format_response(response):
    if not response:
        return "⚠️ No response received."
    wrapped_output = textwrap.fill(response, width=80)
    return f"\n📌 **DeepThought's Response:**\n{wrapped_output}\n"

def get_image_file_from_user() -> str | None:
    while True:
        file_path = input("Enter the path to the image file or hit enter to skip image input: ")
        if not file_path:
            return None
        if os.path.isfile(file_path):
            return file_path
        else:
            print("Invalid file path. Please try again.")

def get_text_input_from_user() -> str:
    return input("Enter text for DeepThought or type 'exit' or 'quit' to end: ")

def build_conversation_no_image(user_input_text: str) -> dict:
    return {
        "role": "user",
        "content": [{"type": "text", "text": user_input_text}]
    }

def build_conversation(user_input_text: str, image: str) -> dict:
    return {
        "role": "user",
        "content": [
            {"type": "text", "text": user_input_text},
            {"type": "image", "image": Image.open(image)}
        ]
    }

def log_interaction(role, content):
    history_log.append({
        "timestamp": datetime.now().isoformat(),
        "role": role,
        "content": content
    })

# -------- DeepThought Call --------

def test_talk_to_deepthought(user_convo: dict):
    logging.info("Starting conversation...")
    try:
        conversation.append(user_convo)
        log_interaction(user_convo["role"], user_convo["content"])

        response = get_inference(auth=AUTH_KEY, model=MODEL_LLAMA_VIS, conversation=conversation, streaming=False)
        conversation.append({"role": "assistant", "content": [{"type": "text", "text": response}]})
        log_interaction("assistant", [{"type": "text", "text": response}])

        return response
    except Exception as e:
        logging.error(f"Error occurred: {e}")

# -------- Prompt File Handling --------

def list_prompt_files(directory="prompts") -> list:
    if not os.path.isdir(directory):
        os.makedirs(directory)
        return []
    return [f for f in os.listdir(directory) if f.endswith(".json")]

def load_prompt_from_file(path: str) -> list:
    with open(path, "r") as f:
        prompt = json.load(f)

    for message in prompt:
        for item in message.get("content", []):
            if item["type"] == "image" and isinstance(item["image"], str):
                image_path = item["image"]
                if os.path.isfile(image_path):
                    item["image"] = Image.open(image_path)
                else:
                    print(f"⚠️ Image file not found: {image_path}")
                    item["image"] = None
    return prompt

def select_prompt_file() -> list:
    prompt_files = list_prompt_files()

    print("\n🔧 Choose how you'd like to start the conversation with DeepThought:")
    print("0. 🤖 Default: No preset prompt — just ask your question directly")

    for i, prompt in enumerate(prompt_files, start=1):
        print(f"{i}. 📄 {prompt}")

    choice = input("Enter choice [number]: ")

    try:
        choice = int(choice)
        if choice == 0:
            return [
                {"role": "assistant", "content": [{"type": "text", "text": "You are a helpful assistant."}]}
            ]
        elif 1 <= choice <= len(prompt_files):
            path = os.path.join("prompts", prompt_files[choice - 1])
            return load_prompt_from_file(path)
    except:
        pass

    print("Invalid choice. Using default assistant.")
    return [
        {"role": "assistant", "content": [{"type": "text", "text": "You are a helpful assistant."}]}
    ]

# -------- Logging & Export --------

def save_conversation_to_file():
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"conversation_{timestamp}.json"
    filepath = os.path.join("logs", filename)

    if not os.path.isdir("logs"):
        os.makedirs("logs")

    with open(filepath, "w") as f:
        json.dump(history_log, f, indent=2, default=str)

    print(f"\n💾 Conversation saved to {filepath}")

def export_conversation(log, as_markdown=True):
    if not os.path.isdir("logs"):
        os.makedirs("logs")  # ✅ Make sure the logs/ folder exists

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    ext = "md" if as_markdown else "txt"
    filename = f"conversation_{timestamp}.{ext}"
    filepath = os.path.join("logs", filename)

    lines = []
    for entry in log:
        role = entry.get("role", "unknown").capitalize()
        content = entry.get("content", [])
        text = ""
        for item in content:
            if item["type"] == "text":
                text += item["text"]
            elif item["type"] == "image":
                text += "![Image omitted](image.png)" if as_markdown else "[Image omitted]"
        if as_markdown:
            lines.append(f"### {role}\n\n{text}\n")
        else:
            lines.append(f"{role}:\n{text}\n")

    with open(filepath, "w") as f:
        f.write("\n".join(lines))

    print(f"\n📝 Exported conversation to {filepath}")


def browse_past_conversations():
    logs_dir = "logs"
    if not os.path.isdir(logs_dir):
        print("No past conversations found.")
        return

    files = [f for f in os.listdir(logs_dir) if f.endswith(".json")]
    if not files:
        print("No past conversation logs found.")
        return

    print("\n📚 Available Conversation Logs:")
    for i, f in enumerate(files, start=1):
        print(f"{i}. {f}")

    choice = input("Select a conversation to view [number] or press enter to cancel: ")
    if not choice:
        return

    try:
        index = int(choice) - 1
        if 0 <= index < len(files):
            filepath = os.path.join(logs_dir, files[index])
            with open(filepath, "r") as f:
                log = json.load(f)
                print_conversation_log(log)
        else:
            print("Invalid selection.")
    except Exception as e:
        print(f"Error: {e}")

def print_conversation_log(log):
    print("\n🗂️ Replaying conversation log:\n")
    for entry in log:
        role = entry.get("role", "unknown").capitalize()
        content = entry.get("content", [])
        text = ""
        for item in content:
            if item["type"] == "text":
                text += item["text"] + "\n"
            elif item["type"] == "image":
                text += "[Image content omitted]\n"
        print(f"--- {role} ---\n{text}")

# -------- Main Menu --------

def main_menu():
    print("\n🔷 DeepThought Terminal Interface")
    print("1. 💬 Start new conversation")
    print("2. 📖 View past conversation")
    print("3. 📝 Export last conversation")
    print("4. ❌ Quit")
    return input("Choose an option [1/2/3/4]: ")

def main():
    global conversation, history_log

    while True:
        choice = main_menu()

        if choice == "1":
            history_log = []
            conversation = select_prompt_file()

            # Show loading message
            print("\n🔄 Loading conversation from prompt... please wait.\n")
            progress_bar = tqdm(total=len(conversation), desc="Processing conversation", ncols=100)

             # Temporary list to hold messages to print all at once
            display_log = []

            # 🔁 Auto-run preloaded conversation
            for msg in conversation[:]:  # Copy list
                if msg["role"] == "assistant":
                    # Log preloaded assistant message
                    log_interaction("assistant", msg["content"])
                    assistant_text = "\n".join([item["text"] for item in msg["content"] if item["type"] == "text"])
                    display_log.append(f"\n🤖 Assistant:\n{assistant_text}")

                elif msg["role"] == "user":
                    # Show user message and image indicator
                    user_text = "\n".join([item["text"] for item in msg["content"] if item["type"] == "text"])
                    display_log.append(f"\n👤 User:\n{user_text}")
                    if any(item["type"] == "image" for item in msg["content"]):
                        display_log.append("📸 [Image input provided]")

                    # Send to DeepThought
                    response = test_talk_to_deepthought(msg)
                    formatted = format_response(response)
                    display_log.append(f"\n🤖 Assistant (DeepThought):\n{formatted}")
                
                # Update the progress bar for each message processed
                progress_bar.update(1)

            # Now print everything at once
            print("\n✅ Preloaded conversation:\n")
            for entry in display_log:
                print(entry)
            
            progress_bar.close()

            # 🔄 Begin interactive conversation
            while True:
                user_text = get_text_input_from_user()
                if user_text.lower() in ["exit", "quit"]:
                    break

                image_path = get_image_file_from_user()

                if image_path:
                    convo = build_conversation(user_input_text=user_text, image=image_path)
                else:
                    convo = build_conversation_no_image(user_input_text=user_text)

                print("\n🧠 Sending message to DeepThought... ⏳\n")
                response = test_talk_to_deepthought(convo)
                formatted_response = format_response(response)
                print(formatted_response)

            save_conversation_to_file()

        elif choice == "2":
            browse_past_conversations()

        elif choice == "3":
            export_conversation(history_log, as_markdown=True)

        elif choice == "4":
            print("Goodbye!")
            break

        else:
            print("Invalid selection.")

if __name__ == "__main__":
    main()
