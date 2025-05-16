import os
import json
from datetime import datetime
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import csv
import re
import numpy as np
import textwrap


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
        df["Score"] = pd.to_numeric(df["Score"], errors="coerce")
        df["Preprocessing"] = df["Preprocessing"].str.strip().str.lower()
        df["Prompt Mode"] = df["Prompt Mode"].str.strip().str.lower()

        preprocessings = ["default", "none"]  # Focus on default vs none
        prompt_modes = ["dynamic_rag", "control"]  # Compare RAG vs control
        queries = df["Query"].unique()
        images = df["Image"].unique()

        # Hardcoded colors for images
        custom_colors = ["#fdbb34", "#55fbcf", "#fb5581", "#81f655", "#849bff", "#f0ff1f"]
        # Hardcoded lighter colors for 'none' preprocessing
        lighter_colors = ["#ffcc80", "#a7e0db", "#ff85a1", "#9df398", "#a0c5ff", "#f1f94d"]

        # Create dictionaries for default and none colors
        image_colors = {img: color for img, color in zip(images, custom_colors)}
        light_image_colors = {img: color for img, color in zip(images, lighter_colors)}

        # Grouped mean scores
        grouped = df.groupby(["Query", "Image", "Preprocessing", "Prompt Mode"])["Score"].mean().reset_index()

        # Create output folder
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_folder = f"output_{timestamp}"
        os.makedirs(output_folder, exist_ok=True)

        # Plot one figure for both preprocessing methods (default vs none)
        fig, axes = plt.subplots(2, 1, figsize=(26, 19), sharey=False)
        if not isinstance(axes, (list, np.ndarray)):
            axes = [axes]

        for i, mode in enumerate(prompt_modes):
            ax = axes[i]
            subset = grouped[grouped["Prompt Mode"] == mode]

            bar_width = 0.6  # Bar width for each group of images
            spacing = 0.1 # Add some space between bars
            tick_positions = []
            x_labels = []
            pos = 0

            for q in queries:
                n_images = len(images)
                group_width = (2 * bar_width + spacing) * n_images
                # For each query, we will plot 10 bars: 5 for 'default' and 5 for 'none'
                for j, img in enumerate(images):
                    # Extract scores for 'default' and 'none' preprocessing
                    default_score = subset[(subset["Query"] == q) & 
                                        (subset["Image"] == img) & 
                                        (subset["Preprocessing"] == "default")]["Score"].values
                    none_score = subset[(subset["Query"] == q) & 
                                        (subset["Image"] == img) & 
                                        (subset["Preprocessing"] == "none")]["Score"].values
                    base = pos + j * (2 * bar_width + spacing)

                    # If scores are available, plot them
                    if default_score.size > 0:
                        label = f"{img} -> default"  # Append 'default' to the image name for the label
                        ax.bar(base, default_score[0], width=bar_width, 
                            color=image_colors.get(img, "gray"), label=label if pos == 0 else "")
                        ax.text(base, default_score[0] + 0.1, f"{default_score[0]:.1f}", ha='center', fontsize=14)
                    if none_score.size > 0:
                        label = f"{img} -> none"  # Append 'none' to the image name for the label
                        ax.bar( base + bar_width + spacing, none_score[0], width=bar_width, 
                            color=light_image_colors.get(img, "lightgray"), label=label if pos == 0 else "")
                        ax.text(base + bar_width + spacing, none_score[0] + 0.1, f"{none_score[0]:.1f}", ha='center', fontsize=14)

                # Position and label for each query
                center = pos + (group_width - bar_width - spacing) / 2
                # center = pos + (len(images) * 2 - 1) * bar_width / 2
                wrapped_q = "\n".join(textwrap.wrap(q.strip(), width=25))
                tick_positions.append(center)
                x_labels.append(wrapped_q)
                pos += group_width + 1.0 
                # pos += (len(images) * 2) * (bar_width + spacing) + 0.9

            ax.set_title(f"{mode.upper()} Prompt", fontsize=22)
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(x_labels, rotation=0, ha="center", fontsize=15)
            ax.set_ylabel("Score (out of 5)", fontsize=20)
            ax.set_ylim(0, 6)
            ax.legend(title="Image", fontsize=16, bbox_to_anchor=(1.01, 1), loc='upper left')

        fig.suptitle("Default vs None Preprocessing", fontsize=28)
        plt.tight_layout(rect=[0, 0, 1, 0.99])

        # Save figure
        output_file = os.path.join(output_folder, f"evaluation_{output_folder}_chart.png")
        plt.savefig(output_file, dpi=300)
        plt.close(fig)
        print(f"✅ Saved: {output_file}")


    # def plot_experiment_scores(self, csv_path="logs/experiment_results.csv"):
    #     df = pd.read_csv(csv_path)
    #     df["Score"] = pd.to_numeric(df["Score"], errors="coerce")
    #     df["Preprocessing"] = df["Preprocessing"].str.strip().str.lower()
    #     df["Prompt Mode"] = df["Prompt Mode"].str.strip().str.lower()

    #     preprocessings = ["default", "blur", "canny", "segmented", "threshold", "none"]
    #     prompt_modes = ["dynamic_rag", "control"]
    #     queries = df["Query"].unique()
    #     images = df["Image"].unique()

    #     # Custom colors for images
    #     custom_colors = ["#fdbb34", "#55fbcf", "#fb5581", "#81f655", "#849bff", "#f0ff1f"]
    #     image_colors = {img: color for img, color in zip(images, custom_colors)}

    #     # Grouped mean scores
    #     grouped = df.groupby(["Query", "Image", "Preprocessing", "Prompt Mode"])["Score"].mean().reset_index()

    #     # Create output folder
    #     timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    #     output_folder = f"output_{timestamp}"
    #     os.makedirs(output_folder, exist_ok=True)

    #     # Plot one figure per preprocessing method
    #     for prep in preprocessings:
    #         fig, axes = plt.subplots(1, 2, figsize=(24, 6), sharey=True)
    #         if not isinstance(axes, (list, np.ndarray)):
    #             axes = [axes]

    #         for i, mode in enumerate(prompt_modes):
    #             ax = axes[i]
    #             subset = grouped[
    #                 (grouped["Prompt Mode"] == mode) &
    #                 (grouped["Preprocessing"] == prep)
    #             ]

    #             bar_width = 0.2
    #             tick_positions = []
    #             x_labels = []
    #             pos = 0

    #             for q in queries:
    #                 for j, img in enumerate(images):
    #                     match = subset[
    #                         (subset["Query"] == q) &
    #                         (subset["Image"] == img)
    #                     ]
    #                     score = match["Score"].values[0] if not match.empty else 0
    #                     x = pos + j * bar_width
    #                     ax.bar(
    #                         x, score,
    #                         width=bar_width,
    #                         color=image_colors.get(img, "gray"),
    #                         label=img if pos == 0 else ""
    #                     )
    #                     ax.text(x, score + 0.1, f"{score:.1f}", ha='center', fontsize=8)

    #                 center = pos + ((len(images) - 1) * bar_width) / 2
    #                 wrapped_q = "\n".join(textwrap.wrap(q.strip(), width=25))
    #                 tick_positions.append(center)
    #                 x_labels.append(wrapped_q)
    #                 pos += bar_width * len(images) + 0.4

    #             ax.set_title(f"{mode.upper()} Prompt", fontsize=14)
    #             ax.set_xticks(tick_positions)
    #             ax.set_xticklabels(x_labels, rotation=0, ha="center", fontsize=8)
    #             ax.set_ylabel("Score (out of 5)")
    #             ax.set_ylim(0, 6)
    #             ax.legend(title="Image", fontsize=8)

    #         fig.suptitle(f"{prep.capitalize()} Preprocessing", fontsize=18)
    #         plt.tight_layout(rect=[0, 0, 1, 0.95])

    #         # Save figure
    #         output_file = os.path.join(output_folder, f"{prep}_grouped.png")
    #         plt.savefig(output_file, dpi=300)
    #         plt.close(fig)
    #         print(f"✅ Saved: {output_file}")
