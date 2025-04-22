from source.deepthought import get_inference, get_embedding
from source.conversation import Conversation
from source.utils import Utils
from source.rag import RagDatabase
from source.preprocessor import ImagePreProcessor
import os
from PIL import Image
from tqdm import tqdm

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

    def main_menu(self):
        """ Display the main menu for the user """
        print("\n🔷 DeepThought Terminal Interface")
        print("1. 💬 Start new conversation")
        print("2. 📖 View past conversation")
        print("3. ⚙️ Start new conversation using Static RAG")
        print("4. ⚙️ Start new conversation using Dynamic RAG")
        print("5. 📝 Export last conversation")

        print("6. ❌ Quit")
        return input("Choose an option [1/2/3/4/5/6]: ")


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

                # 🔄 Begin interactive conversation
                while True:
                    # Get user input (text)
                    user_input = self.utils.get_text_input_from_user()
                    if user_input.lower() in ["exit", "quit"]:
                        break

                
                    # Get user image input if provided but DON'T APPEND IT YET
                    user_img = self.utils.get_image_file_from_user()

                    #print(f"\nApply Default Image Pre Processing...\n")
                    # print(f"\nApply Blurring Image Pre Processing...\n")
                    #print(f"\nApply Threshold Image Pre Processing...\n")
                    #print(f"\nApply Segmentation Only Image Pre Processing...\n")
                    #print(f"\nApply Canny Edge Detection Only Image Pre Processing...\n")

                    # Apply image pre porcessing for image
                    #user_pre_processed_img = self.preprocessor.apply_default_preprocessing(user_img)
                    #user_pre_processed_img = self.preprocessor.apply_blurring_only(user_img)
                    #user_pre_processed_img = self.preprocessor.apply_thresholding_only(user_img)
                    #user_pre_processed_img = self.preprocessor.apply_segmentation_only(user_img)
                    #user_pre_processed_img = self.preprocessor.apply_canny_only(user_img)
                

                    # -------------- PASS 1 -------------------------------------------------
                    # first pass - I embed just the user input itself
                    print(f"\n\nStaring Pass 1 of Dynamic RAG...\n\n")

                    # get static rag prompt convo
                    print(f"Getting STATIC RAG...")
                    base_dir = self.utils.get_file_path_of_parent()
                    prompt_path = os.path.join(base_dir, "prompts", "prompt1.json")
                    prompt_convo = self.utils.load_prompt_from_file(prompt_path)

                    # APPEND STATIC RAG : append static rag to conversation
                    self.conversation.append(prompt_convo[0])
                    print(f"Static RAG Added to the conversation.")

                    print(f"Creating embedding for user input...")
                    query_user_input = get_embedding(auth=self.auth_key, model="Meta-Llama-3.1-8B-Instruct", texts=[user_input])
                    print(f"User input embedding complete...")


                    # take vector from the user_input embedding and send to similarity function
                    #print(f"Embedding for user input is:{query_user_input}")
                    # Find the most similar facts using Dynamic RAG
                    print(f"Comparing user embedding to RAG database...")
                    similar_facts_pass_1 = self.rag_db.find_most_similar_facts(query_user_input)
                    print(f"Found the most similar facts to the embedded user query.")

                    # Feed the most similar facts to DeepThought
                    #print("Most similar facts retrieved from RAG:")
                    facts_pass_1 = []
                    for fact in similar_facts_pass_1:
                        facts_pass_1.append(fact[1])
                        #print(f"Fact :{fact[1]}")
                        #print(f"Similarity: {fact[0]} - Fact: {fact[1]}")

                    fact_pass1_convo = {
                    "role": "user",
                    "content": [{"type": "text", "text": f"I have completed an embedding using rag and these facts are the closest related to the user query and image I will provide in the next message: {facts_pass_1}"}]
                    }

                    # APPEND FACTS: Add similar similar facts for pass 1 to the conversation
                    self.conversation.append(fact_pass1_convo)
                    print(f"Facts added to the conversation.")

                    

                    # convert user_input to convo
                    user_convo = {
                        "role": "user",
                        "content": [{"type": "text", "text": f"Here is the usery query if there is an additional message sent after this one it will include an image related to this query, but here is my user query: {user_input}"}]
                    }
                    
                    # APPEND USER QUESTION
                    print(f"User's query added to the conversation.")
                    self.conversation.append(user_convo)

                    # get image data and append to conversation
                    img_convo = None
                    if user_img:
                        img_convo = {
                                    "role": "user",
                                    "content": [{"type": "image", "image": Image.open(user_img)}]
                                    }
                        # APPEND USER IMAGE
                        print(f"User's image added to the conversation.")
                        self.conversation.append(img_convo)

                    

                    # Then take static rag, 15 facts, and image and send it to deepthought

                    # User: static rag
                    # User content Rag
                    #               text: Question + facts
                    #               image: image


                    
                    #print(f"\n\n Printing convo before sending to deepthought: {self.conversation}\n\n")

                    # Get deepthoughts response to rag, facts, question and image
                    print(f"Sending Dynamic RAG conversation to Deepthought...")
                    response_for_rag_facts_user_input = get_inference(auth=self.auth_key, model="Llama-3.2-11B-Vision-Instruct", conversation=self.conversation, streaming=False)
                    self.conversation.append({"role": "assistant", "content": [{"type": "text", "text": f"AI assistants response from pass 1 of RAG: {response_for_rag_facts_user_input}"}]})
                    print(f"Pass 1 of Dynamic Rag Complete")

                    # -------------- PASS 2 -------------------------------------------------
                    print(f"\n\nStaring Pass 2 of Dynamic Rag...\n\n")

                    # Create an embedding vector for the response deepthought generated based on static rag, facts, and user input (txt & images)
                    print(f" Creating an embedding for Deepthoughts Answer...")
                    query_embedding = get_embedding(auth=self.auth_key, model="Meta-Llama-3.1-8B-Instruct", texts=[response_for_rag_facts_user_input])

                    #print(f"Embedding for llm answer from pass 1 is:{query_embedding}")


                    # Find the most similar facts using Dynamic RAG
                    print(f"Comparing Deepthoughts Answer embedding to RAG database...")
                    similar_facts_pass2 = self.rag_db.find_most_similar_facts(query_embedding)
                    print(f"Found the most similar facts to Deepthoughts Answer embedding.")

                    # Step 3: Feed the most similar facts to DeepThought
                    #print("Most similar facts retrieved from RAG:")
                    facts_pass_2 = []
                    for fact in similar_facts_pass2:
                        facts_pass_2.append(fact[1])
                        #print(f"Fact :{fact[1]}")
                        #print(f"Similarity: {fact[0]} - Fact: {fact[1]}")

                    fact_pass2_convo = {
                    "role": "user",
                    "content": [{"type": "text", "text": f"This is a second pass of RAG I've done taking the last answer you gave me and embedding it, comparing it to my RAG database, and grabbing the facts most related to your last response...facts related to your last response:{facts_pass_2}"}]
                    }

                    self.conversation.append(fact_pass2_convo)
                    print(f"Facts added to the conversation.")
                    self.conversation_handler.log_interaction("user", user_convo["content"])
                    if img_convo is not None:
                        self.conversation_handler.log_interaction("user", img_convo["content"])




                    #  SEND PASS 2 TO DEEPTHOUGHT
                    response = get_inference(auth=self.auth_key, model="Llama-3.2-11B-Vision-Instruct", conversation=self.conversation, streaming=False)
                    print(f"Pass 2 of Dynamic RAG complete output pending...\n\n")
                    self.conversation_handler.log_interaction("assistant", [{"type": "text", "text": response}])
                    formatted_response = self.conversation_handler.format_response(response)
                    print(formatted_response)
                    self.conversation.append({"role": "assistant", "content": [{"type": "text", "text": f"AI assistant's response for pass 2 of RAG: {response}"}]})



                self.utils.save_conversation_to_file(self.conversation_handler.history_log)

            elif choice == "5":
                self.utils.export_conversation(self.conversation_handler.history_log, as_markdown=True)

            elif choice == "6":
                print("Goodbye!")
                break

            else:
                print("Invalid selection.")

def setup_db():
    AUTH_KEY = "7ceac845-58cb-4260-a558-e83fcee7d776"
    MODEL_LLAMA_VIS = "Llama-3.2-11B-Vision-Instruct"
    MODEL_LLAMA = "Meta-Llama-3.1-8B-Instruct"
    # Initialize the analyzer with the RAG database path, auth
    rag_db = RagDatabase(db_path="database/rag_db.db")  # Provide the actual path to your RAG database

    rag_db.store_facts_from_json(auth=AUTH_KEY, model=MODEL_LLAMA, json_file="database/rag_db_facts.json")

def main():
    print("Running the Cell Morphology Analyzer...")
    AUTH_KEY = "7ceac845-58cb-4260-a558-e83fcee7d776"
    rag_db_path = "/home/share/groups/mcbs913-2025/image/image_proj/source/database/rag_db.db"
    # Initialize the analyzer with the RAG database path, authorization key, and model name
    analyzer = CellMorphologyAnalyzer(
        rag_db_path=rag_db_path,  # Provide the actual path to the RAG database
        auth_key=AUTH_KEY,         # Provide the actual authorization key
    )    
    # Run the analysis
    analyzer.run()


if __name__ == "__main__":
    main()

   
