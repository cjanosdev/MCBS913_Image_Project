from source.deepthought import get_inference, get_embedding
from source.conversation import Conversation
from source.utils import Utils
from source.rag import RagDatabase
from source.preprocessor import ImagePreProcessor
from collections import defaultdict
from itertools import product
import os
import cv2
from datetime import datetime
from PIL import Image
from tqdm import tqdm
import json
import requests
import time

class CellMorphologyAnalyzer:
    def __init__(self, rag_db_path: str, auth_key: str):
        """ Initializes the CellMorphologyAnalyzer with necessary components """
        self.conversation_handler = Conversation()
        self.conversation = self.conversation_handler.conversation
        self.rag_db = RagDatabase(rag_db_path)
        self.utils = Utils()
        self.auth_key = auth_key
        self.history_log = []
        self.preprocessor = ImagePreProcessor()

        self.image_cache = {}  # 🧠 Image cache for loaded PIL images

        # 📦 Load prompt once
        base_dir = self.utils.get_file_path_of_parent()
        prompt_path = os.path.join(base_dir, "prompts", "prompt1.json")
        self.static_prompt = self.utils.load_prompt_from_file(prompt_path)

    def load_image_once(self, image_path):
        if image_path not in self.image_cache:
            self.image_cache[image_path] = Image.open(image_path)
        return self.image_cache[image_path]

    def evaluate_answer(self, query: str, image_path: str, llm_answer: str) -> (str, int):
        """Evaluate DeepThought's answer against the ground truth."""
        correct_answer = self.utils.get_correct_answer(query, image_path)
        eval_prompt = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": f"""You are an answer evaluator when given a user prompt.

                        It is okay if your answer is more detailed than the 'correct' answer.
                
                """}
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"""Evaluate the following answer for correctness on 5 point scale..

                Query: {query}
                Expected Answer: {correct_answer}
                LLM Answer: {llm_answer}

                Score the LLM's answer on a scale of 1 to 5 for correctness using the format "X/5", where X is a number from 1 to 5 and explain why."""}]
            }]

        result = get_inference(auth=self.auth_key, model="Llama-3.2-11B-Vision-Instruct", conversation=eval_prompt, streaming=False)
        # Parse response
        score = self.utils.extract_score_from_evaluation(result)
        return result, score
    
    def run_control_thought(self, user_input, image_path):
        """
            Executes contorl conversation with Deepthought given
            a user query and image. Returns llm response.
        """
        self.conversation.clear()
        # Append query and image to convo
        convo = self.conversation_handler.build_conversation(user_input_text=user_input, image=image_path)
        self.conversation.append(convo)
        response = get_inference(auth=self.auth_key, model="Llama-3.2-11B-Vision-Instruct", conversation=self.conversation, streaming=False)
        return response
         


    def static_rag(self, prompt_convo: dict) -> str:
        """ Static RAG interaction with the prompt """
        prompt = []
        prompt.append(prompt_convo)
        self.conversation_handler.log_interaction("user", prompt_convo["content"])
        self.conversation.append(prompt_convo)

        print(f"\n\nPrinting the conversation: {self.conversation}\n\n")

        response = get_inference(auth=self.auth_key, model="Llama-3.2-11B-Vision-Instruct", conversation=prompt, streaming=False)
        self.conversation_handler.log_interaction("assistant", [{"type": "text", "text": response}])
        self.conversation.append({"role": "assistant", "content": [{"type": "text", "text": response}]})
        return response
    
    def build_labeled_param_combinations(self, preprocessing_param_sweep):
        """
        Builds 3 labeled parameter sets: mild, moderate, strong.
        Automatically handles single values by converting them to [val, val, val].
        """
        labels = ["mild", "moderate", "strong"]
        flat_grid = {}

        for stage, params in preprocessing_param_sweep.items():
            for key, values in params.items():
                # Wrap non-list values so we can apply logic uniformly
                if not isinstance(values, list):
                    values = [values]

                if len(values) == 1:
                    flat_grid[key] = [values[0]] * 3
                elif len(values) == 2:
                    flat_grid[key] = [values[0], values[1], values[1]]
                else:
                    flat_grid[key] = [values[0], values[len(values) // 2], values[-1]]

        param_sets = []
        for i, label in enumerate(labels):
            param = {k: v[i] for k, v in flat_grid.items()}
            param["param_set_label"] = label  # Add the label to the dict
            if self.preprocessor.is_valid_combination(param):
                param_sets.append(param)
            else:
                print(f"⚠️ Skipping invalid param set: {label} -> {param}")

        return param_sets
    
    def run_ground_truth_pipeline(self, image_path: str, ground_truth_file: str):
        print(f"\n🚀 Running automated ground truth evaluation pipeline...\n")
        with open(ground_truth_file, "r") as f:
            ground_truth_data = json.load(f)

        total_score = 0
        scores = []

        for item in ground_truth_data:
            if os.path.basename(item["image_name"]) != os.path.basename(image_path):
                continue

            query = item["query"]
            print(f"\n🧪 Query: {query}\n")
            print(f"🖼️ Image: {image_path}")

            # Run Dynamic RAG with the query and image
            response = self.run_dynamic_rag(user_input=query, image_path=image_path)

            # Evaluate the result against ground truth
            evaluation_result, score = self.evaluate_answer(query=query, image_path=image_path, llm_answer=response)
            print(f"\n📊 Evaluation:\n{evaluation_result}")

            # Extract score
            total_score += score
           
            scores.append({
            "Timestamp": datetime.now().isoformat(),
            "Query": query,
            "Image": os.path.basename(image_path),
            "LLM Answer": response,
            "Correct Answer": self.utils.get_correct_answer(query, image_path),
            "Score": score,
            "Explanation": evaluation_result,
        })

        print("\n✅ Pipeline Completed.")
        print("\n📋 Summary of Scores:")
        for entry in scores:
            query = entry["Query"]
            score_val = entry["Score"]
            print(f"→ [{score_val}/5] {query}")
        
        valid_scores = [entry["Score"] for entry in scores if entry["Score"] >= 0]
        print(f"\n📈 Final Score: {total_score} / {len(valid_scores) * 5}")
        
          # ✅ Save all results to one CSV
        parent_path = self.utils.get_file_path_of_parent()
        full_path = os.path.join(parent_path, "logs")
        csv_path = os.path.join(full_path, f"experiment_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")

        self.utils.save_experiment_batch(csv_path, scores)

        # ✅ Plot score results at the end
        print("\n📊 Generating plot of LLM correctness scores...\n")
        self.utils.plot_experiment_scores(csv_path=csv_path)

    def run_parameter_sweep(self, image_paths: list[str], ground_truth_file: str):
        prompt_modes = ["dynamic_rag", "control"]
        preprocessing_methods = ["default", "none"]
        top_n_facts_options = [6]

        # preprocessing_param_sweep = {
        #     "gaussian_blur": {
        #         "ksize": [(3, 3), (5, 5), (7, 7)],
        #         "sigmaX": [0.5, 1.0, 2.0],
        #     },
        #     "canny": {
        #         "threshold1": [10, 30, 50],
        #         "threshold2": [60, 100, 150],
        #     },
        #     "in_range": {
        #         "lowerb": [(0, 0, 50), (30, 50, 50), (60, 50, 50)],
        #         "upperb": [(180, 255, 255), (90, 255, 255), (120, 255, 255)],
        #     },
        #     "find_contours": {
        #         "mode": [cv2.RETR_EXTERNAL, cv2.RETR_TREE],
        #         "contour_method": [cv2.CHAIN_APPROX_SIMPLE, cv2.CHAIN_APPROX_NONE],
        #     },
        #     "draw_contours": {
        #         "contourIdx": [-1],  # -1 means draw all contours
        #         "color": [(0, 255, 0), (255, 0, 0), (0, 0, 255)],  # Green, Red, Blue
        #         "thickness": [1, 2, 3],
        #     },
        #     "add_weighted": {
        #         "alpha": [0.6, 0.7, 0.8],  # Weight of original image
        #         "beta": [0.4, 0.3, 0.2],   # Weight of overlay (must match 1 - alpha)
        #         "gamma": [0.0],           # Scalar added to sum (usually 0)
        #     }
        # }

        preprocessing_param_sweep = {
            "gaussian_blur": {
                "ksize": (5, 5),
                "sigmaX": 0,
            },
            "canny": {
                "threshold1": 50,
                "threshold2": 150,
            },
            "in_range": {
                "lowerb": (100, 50, 50),
                "upperb": (140, 255, 255),
                "lowerg": (40, 50, 20),
                "upperg": (80, 255, 255),
                "lowerred1": (0, 70, 50),
                "upperred1": (10, 255, 255),
                "lowerred2": (170, 70, 50),
                "upperred2": (180, 255, 255),
            },
            "find_contours": {
                "mode": cv2.RETR_EXTERNAL,
                "contour_method": cv2.CHAIN_APPROX_SIMPLE,
            },
            "draw_contours": {
                "contourIdx": -1,
                "color": (0, 255, 0),
                "thickness": 2,
            },
            "add_weighted": {
                "alpha": 0.6,
                "beta": 0.4,
                "gamma": 0.0,
            }
        }


        # Build the 3-tier parameter set
        # param_combinations = self.build_labeled_param_combinations(preprocessing_param_sweep)
       

        
        with open(ground_truth_file, "r") as f:
            ground_truth_data = json.load(f)

        all_scores = []
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        for image_path, prompt_mode,  param_values, top_n, preprocessing in product(image_paths, prompt_modes, [preprocessing_param_sweep], top_n_facts_options, preprocessing_methods):
            # print(f"\n{param_values}\n")
            if (preprocessing != "none" and prompt_mode != "control"):
                processed_image_path = self.preprocessor.preprocess_image(image_path, method=preprocessing, **param_values)
            if processed_image_path == "Unknown preprocessing method.":
                return "exit"

            for item in ground_truth_data:
                if os.path.basename(item["image_name"]) != os.path.basename(image_path):
                    continue

                query = item["query"]
                print(f"\n🧪 Query: {query}")
                print(f"🖼️ Image: {image_path}")
                print(f"🧪 Params: {param_values}, Prompt Mode: {prompt_mode}")
                print(f"🧪 Preprocessing: {preprocessing}, Top N of: {top_n}" )

                # 🔁 Run either dynamic or control
                if prompt_mode == "dynamic_rag":
                    response = self.run_dynamic_rag(user_input=query, image_path=processed_image_path, top_n=top_n)
                else:
                    response = self.run_control_thought(user_input=query, image_path=processed_image_path)

                evaluation_result, score = self.evaluate_answer(query=query, image_path=image_path, llm_answer=response)

                all_scores.append({
                    "Timestamp": datetime.now().isoformat(),
                    "Query": query,
                    "Image": os.path.basename(image_path),
                    "Preprocessing": preprocessing,
                    "Params": str(param_values),
                    "Top N": top_n,
                    "Prompt Mode": prompt_mode,
                    "LLM Answer": response,
                    "Correct Answer": self.utils.get_correct_answer(query, image_path),
                    "Score": score,
                    "Explanation": evaluation_result,
                })

        print("\n✅ Pipeline Completed.")
        print("\n📋 Summary of Scores:")
         # Find best parameter set per query
        summary = defaultdict(lambda: defaultdict (lambda:defaultdict (lambda:defaultdict (lambda: defaultdict(list)))))  # query → top_n → param_str → scores

        for entry in all_scores:
            query = entry["Query"]
            top_n = entry.get("Top N", 5)
            param_str = entry["Params"]
            score = entry["Score"]
            method = entry["Preprocessing"]
            mode = entry["Prompt Mode"]
            if isinstance(score, (int, float)) and score >= 0:
                summary[query][top_n][method][mode][param_str].append(score)

        for query in summary:
            print(f"\n🧪 Summary for Query: '{query}'")
            for top_n, method_dict in summary[query].items():
                 for method, mode_dict in method_dict.items():
                    for mode, param_sets in mode_dict.items():
                        print(f"\n🔢 Top N Facts: {top_n}")
                        best_param_str = None
                        best_total = -1
                        for param_str, scores in param_sets.items():
                            total_score = sum(scores)
                            # print(f"   Params: {param_str}")
                            print(f"   ✅ Total Score: {total_score} / {len(scores) * 5}")
                            if total_score > best_total:
                                best_total = total_score
                                best_param_str = param_str

                        accuracy = round((best_total / (len(param_sets[best_param_str]) * 5)) * 100, 2)
                        print(f"   📈 Accuracy: {accuracy}% for {mode}")

        # 📈 Final Mode-Wise Accuracy
        mode_scores = defaultdict(list)
        for entry in all_scores:
            score = entry["Score"]
            if isinstance(score, (int, float)) and score >= 0:
                mode_scores[entry["Prompt Mode"]].append(score)

        print(f"\n📊 OVERALL FINAL SCORES BY PROMPT MODE:")
        mode_accuracies = {}
        for mode in prompt_modes:
            scores = mode_scores.get(mode, [])
            total = sum(scores)
            possible = len(scores) * 5
            accuracy = round((total / possible) * 100, 2) if possible > 0 else 0
            mode_accuracies[mode] = accuracy

            print(f"   🧠 Mode: {mode}")
            print(f"      ✅ Score: {total} / {possible}")
            print(f"      📈 Accuracy: {accuracy}%")

        if "dynamic_rag" in mode_accuracies and "control" in mode_accuracies:
            dynamic_accuracy = mode_accuracies["dynamic_rag"]
            control_accuracy = mode_accuracies["control"]
            delta = round(dynamic_accuracy - control_accuracy, 2)
            better_mode = "dynamic_rag" if delta > 0 else "control"

            print(f"\n🏁 CONTROL VS DYNAMIC RAG COMPARISON:")
            print(f"   🔍 Dynamic RAG Accuracy: {dynamic_accuracy}%")
            print(f"   🧪 Control Accuracy:     {control_accuracy}%")
            print(f"   🎯 {better_mode.upper()} outperformed by {abs(delta)} percentage points")

        all_valid_scores = [
            score for scores in mode_scores.values()
            for score in scores if isinstance(score, (int, float)) and score >= 0
        ]
        overall_total = sum(all_valid_scores)
        overall_possible = len(all_valid_scores) * 5
        overall_accuracy = round((overall_total / overall_possible) * 100, 2) if overall_possible > 0 else 0

        print(f"\n📊 OVERALL COMBINED ACCURACY ACROSS ALL MODES:")
        print(f"   ✅ Total Score: {overall_total} / {overall_possible}")
        print(f"   📈 Accuracy: {overall_accuracy}%")


          # 📊 DEFAULT VS NONE COMPARISON
        print(f"\n🧪 DEFAULT VS NONE PREPROCESSING COMPARISON:")

        def compute_accuracy(scores: list):
            valid = [s for s in scores if isinstance(s, (int, float)) and s >= 0]
            total = sum(valid)
            possible = len(valid) * 5
            return round((total / possible) * 100, 2) if possible > 0 else 0

        method_scores = defaultdict(list)
        for entry in all_scores:
            if isinstance(entry["Score"], (int, float)) and entry["Score"] >= 0:
                method_scores[entry["Preprocessing"]].append(entry["Score"])

        default_scores = method_scores.get("default", [])
        none_scores = method_scores.get("none", [])

        default_accuracy = compute_accuracy(default_scores)
        none_accuracy = compute_accuracy(none_scores)
        diff = round(default_accuracy - none_accuracy, 2)

        print(f"   🧪 Default Accuracy: {default_accuracy}%")
        print(f"   🧪 None Accuracy:    {none_accuracy}%")

        if diff > 0:
            print(f"   ✅ DEFAULT outperformed NONE by {diff} percentage points")
        elif diff < 0:
            print(f"   ❌ NONE outperformed DEFAULT by {abs(diff)} percentage points")
        else:
            print("   ➖ Both methods performed equally.")


        # 📁 Save and plot results
        parent_path = self.utils.get_file_path_of_parent()
        full_path = os.path.join(parent_path, "logs")
        os.makedirs(full_path, exist_ok=True)
        csv_path = os.path.join(full_path, f"sweep_results_{run_id}.csv")

        self.utils.save_experiment_batch(csv_path, all_scores)
        self.utils.plot_experiment_scores(csv_path=csv_path)

        print(f"\n✅ Sweep Completed. Results saved to: {csv_path}")

    def generate_caption(self, image_path: str) -> str:
        """Uses the LLM to generate a caption/summary for the given image."""
        image_prompt = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "You are a microscopy expert. Describe what is visible in this microscopic image. Be discerning of colors and shapes this is very important. Please be very precise when analzying the images. Take care to note the shape and uniformity or lack thereof of cells."},
                    {"type": "image", "image": Image.open(image_path)}
                ]
            }
        ]
        caption = get_inference(auth=self.auth_key, model="Llama-3.2-11B-Vision-Instruct", conversation=image_prompt, streaming=False)
        return caption.strip()

    def infer_required_tags(self, text: str) -> set[str]:
        text = text.lower()
        tags = set()
    
        if any(word in text for word in ["fluorescent", "dye", "staining", "dapi", "gfp", "rhodamine", "hoechst"]):
            tags.update(["fluorescent", "dyes", "staining"])
        if any(word in text for word in ["healthy", "apoptosis", "photobleaching", "blebbing", "death"]):
            tags.update(["apoptosis", "health", "cell_death"])
        if any(word in text for word in ["morphology", "puncta", "filamentous", "fragmented"]):
            tags.update(["morphology", "puncta", "filamentous", "blebbing"])
        if any(word in text for word in ["virus", "invaded", "invasion"]):
            tags.update(["virus", "invasion", "organelle"])
        if any(word in text for word in ["how many", "number", "count"]):
            tags.update(["apoptosis", "count", "nucleus", "lysosome"])
        
        return tags
    
    def get_inference_with_retry(self, conversation, max_retries=3, delay=3):
        for attempt in range(max_retries):
            try:
                return get_inference(
                    auth=self.auth_key,
                    model="Llama-3.2-11B-Vision-Instruct",
                    conversation=conversation,
                    streaming=False
                )
            except requests.exceptions.ConnectionError as ce:
                print(f"[Attempt {attempt+1}] Connection error: {ce}")
            except Exception as e:
                print(f"[Attempt {attempt+1}] General error: {e}")
            
            if attempt < max_retries - 1:
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)
        raise RuntimeError("Failed to get inference after multiple retries.")


    def run_dynamic_rag(self, user_input: str, image_path: str, top_n:int) -> str:
        """
        Executes Dynamic RAG with the given user query and image. Returns the final LLM response.
        """
        # -------------- PASS 1 -------------------------------------------------
        # First pass - embed just the user input itself
        print(f"Staring Pass 1 of Dynamic RAG...")

        # Get static RAG prompt convo
    

        # APPEND STATIC RAG : append static rag to conversation
        self.conversation.clear()
        self.conversation.append(self.static_prompt[0])
  
        # Step 1: Generate image caption (optional but valuable)
        caption = self.generate_caption(image_path)
        # print(f"\n🖼️ Caption: {caption}")

        # Step 2: Embed [query + caption]
        combined_text = f"{user_input} | {caption}"
        query_embedding = get_embedding(auth=self.auth_key, model="Meta-Llama-3.1-8B-Instruct", texts=[combined_text])

        # Infer tags
        query_tags = self.infer_required_tags(user_input)
        caption_tags = self.infer_required_tags(caption)
        required_tags = query_tags.union(caption_tags)


        similar_facts_pass_1 = self.rag_db.find_most_similar_facts(query_embedding, top_n=top_n, required_tags=required_tags)


        # Prepare fact message
        facts_pass_1 = [fact[1] for fact in similar_facts_pass_1]
        formatted_facts = "\n".join([f"- {fact}" for fact in facts_pass_1])

        system_prompt = {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "You are a microscopy analysis expert assisting with evaluation of cell morphology and fluorescent staining. "
                        "Answer each question using both your own knowledge and the provided facts. Be concise, accurate, and image-aware."
                    )
                }
            ]
        }

        # Construct user prompt
        user_prompt = {
        "role": "user",
        "content": [{
            "type": "text",
            "text": (
                f"Relevant facts:\n{formatted_facts}\n\n"
                f"Image summary:\n{caption}\n\n"
                f"Question:\n{user_input}"
            )
        }]
    }

      
        if image_path:
            image_prompt = {
                "role": "user",
                "content": [{"type": "image", "image": Image.open(image_path)}]
            }

        # Assemble conversation
        self.conversation.append(system_prompt)
        self.conversation.append(user_prompt)
        if image_prompt:
            self.conversation.append(image_prompt)

        # Get DeepThought's response (pass 1)
        #print(f"Sending Dynamic RAG conversation to DeepThought...")
        response_pass1 = self.get_inference_with_retry(self.conversation)
        print(f"Pass 1 of Dynamic RAG Complete")
        self.conversation.append({"role": "assistant", "content": [{"type": "text", "text":  response_pass1}]})

        # -------------- PASS 2 -------------------------------------------------
        print(f"Staring Pass 2 of Dynamic RAG...")
        #print(f"Creating an embedding for DeepThought's answer...")
        response_embedding = get_embedding(auth=self.auth_key, model="Meta-Llama-3.1-8B-Instruct", texts=[response_pass1])

        #print(f"Comparing DeepThought's Answer embedding to RAG database...")
        similar_facts_pass2 = self.rag_db.find_most_similar_facts(response_embedding, top_n=top_n)
        #print(f"Found the most similar facts to DeepThought's answer.")

        # Add new facts to conversation
        facts_pass_2 = [fact[1] for fact in similar_facts_pass2]
        refined_fact_text = "\n".join([f"- {fact}" for fact in facts_pass_2])
        refinement_prompt = {
            "role": "user",
            "content": [{
                "type": "text",
                "text": f"Based on your last answer, here are additional facts:\n{refined_fact_text}\n\nNow, revise or expand your answer."
            }]
        }

        self.conversation.append({"role": "assistant", "content": [{"type": "text", "text": response_pass1}]})

        self.conversation.append(refinement_prompt)
        #print(f"Facts added to the conversation.")

        # Log all interactions
        self.conversation_handler.log_interaction("user", user_prompt["content"])
        if image_prompt:
            self.conversation_handler.log_interaction("user", image_prompt["content"])

        # Get DeepThought's response (pass 2)
        #print(f"Sending second pass to DeepThought...")
        #final_response = get_inference(auth=self.auth_key, model="Llama-3.2-11B-Vision-Instruct", conversation=self.conversation, streaming=False)
        final_response = self.get_inference_with_retry(self.conversation)
        print(f"Pass 2 of Dynamic RAG complete output pending...")
        self.conversation_handler.log_interaction("assistant", [{"type": "text", "text": final_response}])
        formatted_response = self.conversation_handler.format_response(final_response)
        print(f"\n{formatted_response}")
        self.conversation.append({"role": "assistant", "content": [{"type": "text", "text": {final_response}}]})

        return final_response


    def main_menu(self):
        """ Display the main menu for the user """
        print("\n🔷 DeepThought Terminal Interface")
        print("1. 💬 Start new conversation")
        print("2. 📖 View past conversation")
        print("3. ⚙️ Start new conversation using Static RAG")
        print("4. ⚙️ Start new conversation using Dynamic RAG")
        print("5. 📝 Export last conversation")
        print("6. Run Automated Evaluation")

        print("7. ❌ Quit")
        return input("Choose an option [1/2/3/4/5/6/7]: ")


    def run(self):
        """ Main function to run the analysis process """
        while True:
            choice = self.main_menu()

            if choice == "1":

                # 🔄 Begin interactive conversation
                while True:
                    user_text = self.utils.get_text_input_from_user()
                    if user_text.lower() in ["exit", "quit"]:
                        break

                    image_path = self.utils.get_image_file_from_user()

                    if image_path:
                        convo = self.conversation_handler.build_conversation(user_input_text=user_text, image=image_path)
                    else:
                        convo = self.conversation_handler.build_conversation(user_input_text=user_text)

                    print("\n🧠 Sending message to DeepThought... ⏳\n")
                    self.conversation.append(convo)
                    self.conversation_handler.log_interaction("user", convo["content"])
                    response = get_inference(auth=self.auth_key, model="Llama-3.2-11B-Vision-Instruct", conversation=self.conversation, streaming=False)
                    self.conversation_handler.log_interaction("assistant", [{"type": "text", "text": response}])
                    self.conversation.append({"role": "assistant", "content": [{"type": "text", "text": response}]})
                    formatted_response = self.conversation_handler.format_response(response)
                    print(formatted_response)

                self.utils.save_conversation_to_file(self.conversation_handler.history_log)

            elif choice == "2":
                self.utils.browse_past_conversations()

            elif choice == "3":
                self.history_log = []
                prompt_convo = Utils.select_prompt_file()
                if prompt_convo[0] == "exit":
                    break # no prompt file loaded

                # Show loading message
                print("\n🔄 Loading conversation from prompt... please wait.\n")
                progress_bar = tqdm(total=len(prompt_convo), desc="Processing conversation", ncols=100)

                # Temporary list to hold messages to print all at once
                display_log = []

                # 🔁 Auto-run preloaded conversation
                # Process conversation with static RAG
                print("\n🔄 Processing conversation using Static RAG...\n")
                response = self.static_rag(prompt_convo[0])
                formatted_response = self.conversation_handler.format_response(response)
                print(formatted_response)
                print("\n✅ Preloaded conversation:\n")
                display_log.append(f"\n🤖 Assistant (DeepThought):\n{formatted_response}")
                    
                # Update the progress bar for each message processed
                progress_bar.update(1)
                progress_bar.close()
            
                # 🔄 Begin interactive conversation
                while True:
                    user_text = self.utils.get_text_input_from_user()
                    if user_text.lower() in ["exit", "quit"]:
                        break

                    image_path = self.utils.get_image_file_from_user()

                    if image_path:
                        convo = self.conversation_handler.build_conversation(user_input_text=user_text, image=image_path)
                    else:
                        convo = self.conversation_handler.build_conversation(user_input_text=user_text)

                    print("\n🧠 Sending message to DeepThought... ⏳\n")
                    self.conversation.append(convo)
                    self.conversation_handler.log_interaction("user", convo["content"])
                    response = get_inference(auth=self.auth_key, model="Llama-3.2-11B-Vision-Instruct", conversation=self.conversation, streaming=False)
                    self.conversation_handler.log_interaction("assistant", [{"type": "text", "text": response}])
                    self.conversation.append({"role": "assistant", "content": [{"type": "text", "text": response}]})
                    formatted_response = self.conversation_handler.format_response(response)
                    print(formatted_response)

                self.utils.save_conversation_to_file(self.conversation_handler.history_log)

            elif choice == "4":
                # Process conversation with Dynamic RAG
                print(f"\n🔄 Processing conversation using Dynamic RAG...\n")
                while True:
                    user_input = self.utils.get_text_input_from_user()
                    if user_input.lower() in ["exit", "quit"]:
                        break

                    user_img = self.utils.get_image_file_from_user()

                    # Run the dynamic RAG pipeline
                    response = self.run_dynamic_rag(user_input=user_input, image_path=user_img)

                    # Evaluate and print result
                    evaluation = self.evaluate_answer(query=user_input, image_path=user_img, llm_answer=response)
                    print("\n📊 Evaluation of DeepThought's Answer:\n")
                    print(evaluation)

                self.utils.save_conversation_to_file(self.conversation_handler.history_log)

            elif choice == "5":
                self.utils.export_conversation(self.conversation_handler.history_log, as_markdown=True)

            elif choice == "6":
                base_dir = self.utils.get_file_path_of_parent()
                file_name = os.path.join(base_dir, "source/database/ground_truth.json")
                #self.run_ground_truth_pipeline("/home/share/groups/mcbs913-2025/image/image_data_sets/Fluorescence/NucleusLysosomeStain.png", file_name)
                # image_path_list = [
                # "/home/share/groups/mcbs913-2025/image/image_data_sets/Fluorescence/NucleusLysosomeStain.png",
                # ]
                image_path_list = [
                    "/home/share/groups/mcbs913-2025/image/image_data_sets/Fluorescence/NucleusLysosomeStain.png",
                    "/home/share/groups/mcbs913-2025/image/image_data_sets/Fluorescence/ERswelling.png",
                    "/home/share/groups/mcbs913-2025/image/image_data_sets/Fluorescence/NucleusandMitoStain.png",
                    "/home/share/groups/mcbs913-2025/image/image_data_sets/Fluorescence/VirusInvasion.png"]
                parameter_sweep = self.run_parameter_sweep(image_path_list, file_name)
                if parameter_sweep == "exit":
                    print("Goodbye!")
                    break

            elif choice == "7":
                print("Goodbye!")
                break

            else:
                print("Invalid selection.")

def setup_db():
    AUTH_KEY = "7ceac845-58cb-4260-a558-e83fcee7d776"
    MODEL_LLAMA_VIS = "Llama-3.2-11B-Vision-Instruct"
    MODEL_LLAMA = "Meta-Llama-3.1-8B-Instruct"
    # Initialize the analyzer with the RAG database path, auth
    #rag_db = RagDatabase(db_path="database/rag_db_tagged.db")  # Provide the actual path to your RAG database
    #rag_db = RagDatabase(db_path="database/rag_db.db")
    #rag_db2 = RagDatabase(db_path="database/rag_db_2.db")

    #rag_db2.store_facts_from_json(auth=AUTH_KEY, model=MODEL_LLAMA, json_file="database/rag_db_2.json")
    #rag_db.store_facts_from_json(auth=AUTH_KEY, model=MODEL_LLAMA, json_file="database/rag_db_facts_tagged.json")
    #rag_db.store_facts_from_json(auth=AUTH_KEY, model=MODEL_LLAMA, json_file="database/rag_db_facts.json")

def main():
    print("Running the Cell Morphology Analyzer...")
    AUTH_KEY = "7ceac845-58cb-4260-a558-e83fcee7d776"
    #rag_db_path = "/home/share/groups/mcbs913-2025/image/image_proj/source/database/rag_db.db"
    rag_db_path = "/home/share/groups/mcbs913-2025/image/image_proj/source/database/rag_db_2.db"
    # Initialize the analyzer with the RAG database path, authorization key, and model name
    analyzer = CellMorphologyAnalyzer(
        rag_db_path=rag_db_path,  # Provide the actual path to the RAG database
        auth_key=AUTH_KEY,         # Provide the actual authorization key
    )    
    # Run the analysis
    analyzer.run()


if __name__ == "__main__":
    #setup_db()
    main()

   
