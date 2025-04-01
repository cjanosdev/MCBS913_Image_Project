from textual.app import App, ComposeResult
from textual.widgets import (
    Header,
    Footer,
    Input,
    Button,
    ProgressBar,
    Label,
    OptionList,
    MarkdownViewer,
    Tree
)
from textual.containers import Vertical, ScrollableContainer, Horizontal
from textual.events import Key
from deepthought import *
from PIL import Image
import os
import json
import asyncio
from datetime import datetime

AUTH_KEY = "7ceac845-58cb-4260-a558-e83fcee7d776"
MODEL_LLAMA_VIS = "Llama-3.2-11B-Vision-Instruct"

conversation = []
history_log = []

def build_conversation(user_input_text: str, image_path: str | None = None) -> dict:
    content = [{"type": "text", "text": user_input_text}]
    if image_path and os.path.isfile(image_path):
        content.append({"type": "image", "image": Image.open(image_path)})
    return {"role": "user", "content": content}

def test_talk_to_deepthought(user_convo: dict):
    conversation.append(user_convo)
    history_log.append({
        "timestamp": datetime.now().isoformat(),
        "role": user_convo["role"],
        "content": user_convo["content"]
    })
    try:
        response = get_inference(auth=AUTH_KEY, model=MODEL_LLAMA_VIS, conversation=conversation, streaming=False)
        conversation.append({"role": "assistant", "content": [{"type": "text", "text": response}]})
        history_log.append({
            "timestamp": datetime.now().isoformat(),
            "role": "assistant",
            "content": [{"type": "text", "text": response}]
        })
        return response
    except Exception as e:
        return f"⚠️ Error: {e}"

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
                    item["image"] = None
    return prompt

class DeepThoughtTUI(App):
    CSS_PATH = "deepthought.tcss"
    BINDINGS = [("q", "quit", "Quit")]

    def __init__(self):
        super().__init__()
        self.markdown_log = "🔄 Select a prompt or start typing.\n\n"

    def compose(self) -> ComposeResult:
        yield Header()
        yield Horizontal(
            Vertical(
                Label("Choose a prompt to preload (optional):"),
                OptionList("(None) Start from scratch", *list_prompt_files(), id="prompt_selector"),
                Label("Prompt Text:"),
                Input(placeholder="Ask anything",id="prompt_input"),
                # Label("Image Path (optional):"),
                # Input(placeholder="Input path to image file", id="image_input"),
                Label("Select Image (optional):"),
                Tree("📁 Current Directory", id="image_tree"),
                Label("📸 No image selected.", id="image_status"),
                Label("", id="toast_label"),  # <- toast placeholder
                Horizontal(
                    Button("Send", id="send_button"),
                    Button("Exit", id="exit_button", variant="error"),
                    Button("Save", id="save_button", variant="primary"),
                    id="button_bar"
                ),
                ProgressBar(id="progress_bar", total=100),
                id="controls"
            ),
            MarkdownViewer("Loading...", show_table_of_contents=False, id="response_log"),
            id="main_layout"
        )
        yield Footer()


    def update_markdown(self):
        viewer = self.query_one("#response_log", MarkdownViewer)
        viewer.document.update(self.markdown_log)

    def append_to_log(self, text: str):
        self.markdown_log += text + "\n\n"
        self.update_markdown()

    async def show_local_toast(self, message: str, style: str = "yellow"):
        label = self.query_one("#toast_label", Label)
        label.update(message)
        label.styles.color = style

        await asyncio.sleep(0)
        await asyncio.sleep(3)  # Then wait without blocking

        label.update("")


    def on_mount(self) -> None:
        self.query_one("#progress_bar", ProgressBar).progress = 0
        self.query_one("#prompt_selector", OptionList).focus()
        tree = self.query_one("#image_tree", Tree)
        self.populate_tree(tree.root, os.getcwd())
        tree.root.expand()
        self.update_markdown()

    def on_tree_node_selected(self, event: Tree.NodeSelected[str]):
        node = event.node
        if node.data and os.path.isfile(node.data):
            self.selected_image_path = node.data
            self.query_one("#image_status", Label).update(
                f"📸 Selected: {os.path.basename(node.data)}"
            )

    def populate_tree(self, node, path):
        try:
            for entry in sorted(os.listdir(path)):
                full_path = os.path.join(path, entry)
                if os.path.isdir(full_path):
                    branch = node.add(entry, expand=False)
                    self.populate_tree(branch, full_path)
                elif entry.lower().endswith((".png", ".jpg", ".jpeg", ".gif")):
                    node.add_leaf(entry, data=full_path)
        except PermissionError:
            pass  # Skip folders without access



    async def on_button_pressed(self, event):
        if event.button.id == "send_button":
            await self.send_prompt()
        elif event.button.id == "exit_button":
            await self.action_quit()

    def on_input_submitted(self, message: Input.Submitted):
        self.send_prompt()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        selected_label = event.option.prompt
        self.handle_prompt_selection(selected_label)

    def handle_prompt_selection(self, selected_label: str):
        if selected_label == "(None) Start from scratch":
            self.markdown_log = "🧠 Starting fresh...\n\n"
            self.markdown_log = "No preset prompt — just ask your question directly in the Prompt Text Input box...\n\n"
            conversation.clear()
            self.update_markdown()
        else:
            path = os.path.join("prompts", selected_label)
            print(self.markdown_log)

            self.markdown_log = f"📂 **Loading prompt:** `{selected_label}`\n\n"
            print(self.markdown_log)

            self.update_markdown()
            self.load_preloaded_conversation(load_prompt_from_file(path))
            self.update_markdown()


    async def send_prompt(self):
        input_widget = self.query_one("#prompt_input", Input)
        text = input_widget.value.strip()
        image_path = getattr(self, "selected_image_path", None)

        if not text:
            await self.show_local_toast("⚠️ Prompt text is required.", style="yellow")
            input_widget.focus()
            return

        if not image_path:
            await self.show_local_toast("⚠️ No image selected. Sending anyway...", style="lightblue")

        self.append_to_log(f"### 👤 You\n{text}")
        if image_path:
            self.append_to_log(f"📎 *Image attached:* `{image_path}`")

        convo = build_conversation(text, image_path)
        response = test_talk_to_deepthought(convo)
        self.append_to_log(f"### 🤖 DeepThought\n{response}")

        input_widget.value = ""
        self.selected_image_path = None
        self.query_one("#image_status", Label).update("📸 No image selected.")


    def load_preloaded_conversation(self, prompt_messages: list):
        bar = self.query_one("#progress_bar", ProgressBar)
        bar.progress = 0
        steps = len(prompt_messages)
        step = 0

        for msg in prompt_messages:
            if msg["role"] == "assistant":
                text = "\n".join([item["text"] for item in msg["content"] if item["type"] == "text"])
                self.append_to_log(f"### 🤖 Assistant\n{text}")
            elif msg["role"] == "user":
                text = "\n".join([item["text"] for item in msg["content"] if item["type"] == "text"])
                self.append_to_log(f"### 👤 User\n{text}")
                if any(item["type"] == "image" for item in msg["content"]):
                    self.append_to_log("📸 *[Image input provided]*")

            step += 1
            bar.progress = int((step / steps) * 100)

        self.append_to_log("✅ *Preloaded conversation loaded. You may continue typing below.*")

if __name__ == "__main__":
    os.makedirs("logs", exist_ok=True)
    os.makedirs("prompts", exist_ok=True)
    DeepThoughtTUI().run()
