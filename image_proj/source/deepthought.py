import base64
import getpass
import io
import json
import numpy as np
import requests
import logging
from datetime import datetime
from PIL import Image
from typing import Callable, Dict, Iterator, List, Union, Tuple
import urllib3

# Suppress SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


API_URL = "https://dtcontroller.sr.unh.edu:4242/api"

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DTJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        # Serialize image as base64 encoded PNG bytes
        if isinstance(obj, Image.Image):
            out_bytes = io.BytesIO()
            obj.save(out_bytes, format="PNG")
            img_base64 = base64.b64encode(out_bytes.getvalue()).decode("utf-8")
            logger.debug(f"Serialized image to base64: {img_base64[:30]}...")  # Log first 30 characters of base64 string
            return img_base64

        return super().default(obj)


def _inference_gen(auth: str, model: str, temperature: float, conversation: List[Dict[str, str]], streaming: bool, user: str=getpass.getuser()) -> Iterator[str]:
    """ Internal: Get inference result generator from server """
    #logger.info(f"Sending inference request to model: {model}")
    send_data = { "auth": auth, "action": "inference_streaming" if streaming else "inference", "model": model, "user": user, "temperature": temperature, "conversation": conversation }
    recv_data = {}


    # logger.info(f"Request data: {send_data}")
    # logger.debug(f"Request data: {send_data}")

    try:
        with requests.post(API_URL, data=json.dumps(send_data, cls=DTJSONEncoder), headers={"Content-Type": "application/json"}, stream=True, verify=False) as response:
            #logger.info(f"reponse is {response}\n")
            if response.status_code != 200:
                raise Exception(f"Status Code {response.status_code}")

            buffer = ""
            for chunk in response.iter_content(chunk_size=256):
                #logger.info(f"Received chunk: {chunk[:50]}...")
                buffer += chunk.decode("utf-8")
               # logger.info(f"\nbuffer")
                try:
                   # logger.info(f"Buffer so far: {buffer}")
                    recv_data = json.loads(buffer)
                    buffer = ""

                    if "error" not in recv_data:
                        raise Exception(f"Unknown error: {recv_data}")

                    if recv_data["error"] is not None:
                        raise Exception(recv_data["error"])

                    if not streaming or not recv_data["final"]:
                        logger.debug(f"Received partial response: {recv_data.get('output', '')}")
                        yield recv_data["output"]

                except json.JSONDecodeError as e:
                    pass
                    #logger.info(f"JSON DecodeError: {e} - Buffer: {buffer}")
    except Exception as e:
        logger.error(f"Request failed with error: {e}")
        raise

    if recv_data == {}:
        raise Exception("Server returned no data")


def _get_recv_data(auth: str, send_data: dict) -> dict:
    """ Internal: Prepare and send request """
    #logger.info(f"Preparing data for action: {send_data['action']}")
    recv_data = {}

    try:
        response = requests.post(API_URL, json=send_data, verify=False)
        if response.status_code != 200:
            raise Exception(f"Status Code {response.status_code}")

        recv_data = response.json()
        recv_data = json.loads(recv_data) if isinstance(recv_data, str) else recv_data

        if recv_data == {}:
            raise Exception("Server returned no data")
        
        if "error" not in recv_data:
            raise Exception(f"Unknown error: {recv_data}")

        if recv_data["error"] is not None:
            raise Exception(recv_data["error"])

        logger.debug(f"Received response: {recv_data}")
    except Exception as e:
        logger.error(f"Failed to get data: {e}")
        raise

    return recv_data


def get_inference(auth: str, model: str, conversation: List[Dict[str, str]], streaming: Union[bool, Callable], temperature: float=0.3, user: str=getpass.getuser()) -> Union[str, Iterator[str]]:
    """ API: Retrieve inference given conversation from server """
    #logger.info("Getting inference...")

    if streaming is None or streaming == False:
        #logger.info("Non-streaming mode selected.")
        return next(_inference_gen(auth, model, temperature, conversation, False, user))

    if streaming == True:
        #logger.info("Streaming mode selected.")
        return _inference_gen(auth, model, temperature, conversation, True, user)

    response = ""
    logger.info("Callback mode selected, processing partial responses...")
    for partial in _inference_gen(auth, model, temperature, conversation, True, user):
        response += partial
        streaming(partial)

    return response


def get_embedding(auth: str, model: str, texts: List[str], weights: List[float]=None, user: str=getpass.getuser()) -> List[float]:
    """ API: Get weighted aggregated embedding vector for the given text(s) from server """
    #logger.info(f"Getting embedding for model: {model}")

    if weights is None:
        weights = [ 1 ] * len(texts)
    
    embeddings_data = [{ "content": texts[x], "weight": weights[x] } for x in range(len(texts))]
    send_data = { "auth": auth, "action": "embedding", "model": model, "user": user, "streaming": False, "conversation": embeddings_data }
    recv_data = _get_recv_data(auth, send_data)

    embedding_bytes = base64.b64decode(recv_data["output"])
    logger.debug(f"Received embedding: {embedding_bytes[:30]}...")  # Log first 30 bytes of embedding
    return np.frombuffer(embedding_bytes, dtype=np.float32)


def get_rag(auth: str, model: str, text: str, documents: List[str], count: int, user: str=getpass.getuser()) -> List[Dict[str, str]]:
    """ API: Get closest RAG hits against specified documents for the given text(s) from server """
    logger.info(f"Getting RAG for text: {text}")
    embeddings_data = [{ "content": text, "weight": 1 }]
    send_data = { "auth": auth, "action": "rag", "model": model, "user": user, "streaming": False, "conversation": embeddings_data, "rag_documents": documents, "rag_count": count }
    recv_data = _get_recv_data(auth, send_data)

    return recv_data["output"]


def get_documents(auth: str, user: str=getpass.getuser()) -> List[Dict[str, str]]:
    """ API: Get available RAG documents """
    logger.info("Getting available RAG documents...")
    send_data = { "auth": auth, "action": "documents", "user": user }
    recv_data = _get_recv_data(auth, send_data)
    
    return recv_data["output"]


def get_models(auth: str, user: str=getpass.getuser()) -> Dict[str, str]:
    """ API: Get available models """
    logger.info("Getting available models...")
    send_data = { "auth": auth, "action": "models", "user": user }
    recv_data = _get_recv_data(auth, send_data)
    
    return recv_data["output"]


def get_activity(auth: str, activity_range: Tuple[datetime, datetime], user: str=getpass.getuser()) -> dict:
    """ API: Get activity history """
    logger.info(f"Getting activity history for range: {activity_range}")
    seconds_range = tuple(date.timestamp() for date in activity_range)
    send_data = { "auth": auth, "action": "activity", "activity_range": seconds_range, "user": user }
    recv_data = _get_recv_data(auth, send_data)
    
    return recv_data["output"]


def get_status(auth: str, user: str=getpass.getuser()) -> dict:
    """ API: Get controller status """
    logger.info("Getting controller status...")
    send_data = { "auth": auth, "action": "status", "user": user }
    recv_data = _get_recv_data(auth, send_data)
    
    return recv_data["output"]



def get_status(auth: str, user: str=getpass.getuser()) -> dict:
    """ API: Get controller status """
    send_data = { "auth": auth, "action": "status", "user": user }
    recv_data = _get_recv_data(auth, send_data)
    
    return recv_data["output"]