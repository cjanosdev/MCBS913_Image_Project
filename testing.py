from deepthought import *
from PIL import Image
import time
import logging

logging.basicConfig(level=logging.INFO)

AUTH_KEY = "7ceac845-58cb-4260-a558-e83fcee7d776"
MODEL_LLAMA_VIS = "Llama-3.2-11B-Vision-Instruct"
MODEL_LLAMA = "Meta-Llama-3.1-8B-Instruct"
#MODEL_LLAMA_VIS ='Llama-3.2-11B-Vision-Instruct'


def convert_image_to_base64():
    """Converts the image to a base64-encoded string"""
    with Image.open("sample_image.png") as img:
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_base64

def test_talk_to_deepthought():
    """ Sends an image and request to DeepThought and prints the inference result """
    logging.info("Starting conversation...")
    
    # TODO: Add this back to conversation variable after getting just text to work.
    #{ "type": "image", "image": Image.open("sample_image.png") }
    
    
    conversation = [
        { 
            "role": "user", 
            "content": [
                { "type": "text", "text": "Please define what a dog is" },
                
            ]
        }
    ]

    logging.info("Sending request to server...")
    try:
        response = get_inference(auth=AUTH_KEY, model=MODEL_LLAMA, conversation=conversation, streaming=False)
        logging.info("Received response, streaming...")

        # if stream set to false
        print(response)
        
        # If stream set to true
        # for response in response_generator:
        #     print(response)
    except Exception as e:
        logging.error(f"Error occurred: {e}")

def get_available_models():
    """ Calls get_models to retrieve and print the available models """
    try:
        logging.info("Fetching available models...")
        models = get_models(AUTH_KEY)
        logging.info(f"Available models: {models}\n")
    except Exception as e:
        logging.error(f"Error fetching models: {e}")

# Run the function
get_available_models()  # Fetch and print the available models
test_talk_to_deepthought()