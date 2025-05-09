import os
import json
from datetime import datetime
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import csv
import re

class Utils:

    @staticmethod
    def get_file_path_of_parent() -> str:
        """ Return the parent directory of the current script file """
        # Get the absolute path of the current script
        current_file_path = os.path.abspath(__file__)
        
        # Get the parent directory of the current file
        parent_file_path = os.path.abspath(os.path.join(current_file_path, "..", ".."))
        # print(f"\n\n Parent file path is: {parent_file_path}\n\n")
        
        return parent_file_path


    @staticmethod
    def get_image_file_from_user() -> str | None:
        """ Prompt the user for an image file and return the path """
        while True:
            file_path = input("Enter the path to the image file or hit enter to skip image input: ")
            if not file_path:
                return None
            if os.path.isfile(file_path):
                return file_path
            else:
                print("Invalid file path. Please try again.")

    @staticmethod
    def get_text_input_from_user() -> str:
        """ Get text input from the user """
        return input("Enter text for DeepThought or type 'exit' or 'quit' to end: ")

    @staticmethod
    def list_prompt_files(directory="prompts") -> list:
        """ List all prompt files in the prompts directory """
        # Ensure the directory is correctly constructed relative to the script location
        base_dir = Utils.get_file_path_of_parent()
        full_path = os.path.join(base_dir, directory)  # Go one level up and then to the prompts folder
        print(f"\nlist prompt files path: {full_path}\n")
        # Check if the 'prompts' directory exists
        if not os.path.isdir(full_path):
            os.makedirs(full_path)  # Create the 'prompts' directory if it doesn't exist
            return []

        return [f for f in os.listdir(full_path) if f.endswith(".json")]
    
    @staticmethod
    def select_prompt_file() -> list:
        prompt_files = Utils.list_prompt_files()

        for i, prompt in enumerate(prompt_files, start=1):
            print(f"{i}. 📄 {prompt}")
        
        choice = input("Enter choice [number]: ")

        try:
            choice = int(choice)
            print(f"\nChoice is {choice}\n")
            if 1 <= choice <= len(prompt_files):
                # Construct the correct path to the prompt file
                base_dir = Utils.get_file_path_of_parent()
                full_path = os.path.join(base_dir, "prompts", prompt_files[choice - 1])  # Go one level up and then to the prompts folder
                print(f"\nSelected prompt file path: {full_path}\n")
                
                # Load the selected prompt from the file
                return Utils.load_prompt_from_file(full_path)
            else:
                print("Choice is out of range.")
        except ValueError:
            # Catching the ValueError explicitly if the input isn't a valid integer
            print("Invalid input. Please enter a valid number.")
        
        print("Invalid choice. Exiting...")
        return ["exit"]

    @staticmethod
    def load_prompt_from_file(path: str) -> list:
        """ Load a prompt from a file and process any image references """
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

    @staticmethod
    def log_interaction(history_log, role: str, content: dict):
        """ Log the conversation interaction for later retrieval or export """
        history_log.append({
            "timestamp": datetime.now().isoformat(),
            "role": role,
            "content": content
        })

    @staticmethod
    def save_conversation_to_file(history_log):
        """ Save the conversation to a file """
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"conversation_{timestamp}.json"

        base_dir = Utils.get_file_path_of_parent()
        logs_dir = os.path.join(base_dir, "logs")
        # print(f"\n saving convo to file at location: {logs_dir}\n")
        filepath = os.path.join(logs_dir, filename)

        if not os.path.isdir(logs_dir):
            os.makedirs(logs_dir)

        with open(filepath, "w") as f:
            json.dump(history_log, f, indent=2, default=str)

        print(f"\n💾 Conversation saved to {filepath}")

    @staticmethod
    def export_conversation(history_log, as_markdown=True):
        """ Export the conversation to a file in markdown or text format """
        base_dir = Utils.get_file_path_of_parent()
        logs_dir = os.path.join(base_dir, "logs")
        print(f"\n export convo to file at location: {logs_dir}\n")

        if not os.path.isdir(logs_dir):
            os.makedirs(logs_dir)

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        ext = "md" if as_markdown else "txt"
        filename = f"conversation_{timestamp}.{ext}"
        filepath = os.path.join(logs_dir, filename)

        lines = []
        print(f"\n\n history log in export is {history_log}\n\n")
        for entry in history_log:
            role = entry.get("role", "unknown").capitalize()
            content = entry.get("content", [])
            text = ""

            for item in content:
                if item["type"] == "text":
                    # Iterate over the list of strings and concatenate them
                    if isinstance(item["text"], list):
                        text += "\n".join(item["text"])  # Concatenate all strings in the list
                    else:
                        text += item["text"]  # If it's a single string, just append it
                elif item["type"] == "image":
                    text += "![Image omitted](image.png)" if as_markdown else "[Image omitted]"
            
            if as_markdown:
                print(f"{role}: {text}")
                lines.append(f"### {role}\n\n{text}\n")
            else:
                lines.append(f"{role}:\n{text}\n")

        with open(filepath, "w") as f:
            f.write("\n".join(lines))

        print(f"\n📝 Exported conversation to {filepath}")

    @staticmethod
    def browse_past_conversations():
        """ Browse and view past conversation logs """
        base_dir = Utils.get_file_path_of_parent()
        logs_dir = os.path.join(base_dir, "logs")
        print(f"\n browse past convo at file location: {logs_dir}\n")
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
                    Utils.print_conversation_log(log)
            else:
                print("Invalid selection.")
        except Exception as e:
            print(f"Error: {e}")

    @staticmethod
    def print_conversation_log(log):
        """ Print a past conversation log """
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

    def get_correct_answer(self, query: str, image_path: str) -> str:
        base_dir = Utils.get_file_path_of_parent()
        file_name = os.path.join(base_dir, "source/database/ground_truth.json")
        with open(file_name) as f:
            data = json.load(f)
        for item in data:
            if item["query"] == query and os.path.basename(image_path) == item["image_name"]:
                return item["expected_answer"]
        return "No ground truth found."

    def log_experiment_result(self, output_path, query, image_name, llm_answer, correct_answer, score, explanation):
        file_exists = os.path.isfile(output_path)
        with open(output_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(["Timestamp", "Query", "Image", "LLM Answer", "Correct Answer", "Score", "Explanation"])
            writer.writerow([
                datetime.now().isoformat(),
                query,
                image_name,
                llm_answer,
                correct_answer,
                score,
                explanation
            ])
    def save_experiment_batch(self, output_path: str, results: list):

        if not results:
            print("No results to write.")
            return

        with open(output_path, mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)

        print(f"✅ Batch CSV saved to: {output_path}")

    def extract_score_from_evaluation(self, response: str) -> int:
        # Match variations like "3/5", "3 out of 5", "3 / 5", etc.
        match = re.search(r"\b([1-5])\s*(?:/|out of)\s*5\b", response, re.IGNORECASE)
        return int(match.group(1)) if match else -1

    def plot_experiment_scores(self, csv_path="logs/experiment_results.csv"):
        df = pd.read_csv(csv_path)

         # ✅ Convert score column to numeric explicitly
        df["Score"] = pd.to_numeric(df["Score"], errors="coerce")
        # ✅ Prepare output path for PNG
        png_path = os.path.splitext(csv_path)[0] + ".png"

        # Optional: Shorten long questions for cleaner axis labels
        df["ShortQuery"] = df["Query"].apply(lambda q: q if len(q) < 60 else q[:57] + "...")


        # Plot as bar chart
        plt.figure(figsize=(12, 6))
        plt.bar(df["ShortQuery"], df["Score"], color='skyblue')
        plt.title("LLM Correctness Score by Question")
        plt.xlabel("Question")
        plt.ylabel("Score (out of 5)")
        plt.ylim(0, 5.5)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

         # ✅ Save plot to same directory as CSV
        plt.savefig(png_path)
        print(f"📁 Plot saved to: {png_path}")
