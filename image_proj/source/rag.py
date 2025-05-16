import sqlite3
import numpy as np
import json
from source.deepthought import get_embedding  # Assuming this function is available for getting embeddings
import os

class RagDatabase:
    def __init__(self, db_path: str):
        """ Initialize the RAG database, creating it if it doesn't exist """
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)  # Create or connect to the database
        self.cursor = self.conn.cursor()
        
        # Ensure the rag_facts table exists
        self._create_table()

    def _create_table(self):
        """ Creates the rag_facts table if it doesn't already exist """
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS rag_facts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                embedding_vector BLOB NOT NULL,
                fact_chunk TEXT NOT NULL,
                tags TEXT
            )
        ''')
        self.conn.commit()

    def store_fact(self, auth, model, fact_chunk: str, tags: str):
        """ Store a fact and its embedding in the RAG database """
        # Get the embedding for the fact
        embedding = get_embedding(auth, model, [fact_chunk])
        embedding_blob = embedding.tobytes()  # Convert the embedding to a binary format
        tags_str = ",".join(tags) if tags else ""
        
        # Insert the fact and its embedding into the database
        self.cursor.execute('''
            INSERT INTO rag_facts (embedding_vector, fact_chunk, tags)
            VALUES (?, ?, ?)
        ''', (embedding_blob, fact_chunk, tags_str))
        self.conn.commit()

        print(f"Stored fact: {fact_chunk}")

    def store_facts_from_json(self, auth, model, json_file: str):
        """ Load facts from a JSON file and store them in the RAG database """
        if not os.path.isfile(json_file):
            print(f"⚠️ JSON file {json_file} not found.")
            return

        with open(json_file, 'r') as f:
            facts = json.load(f)  # Load the JSON data

        # Store each fact in the RAG database
        for fact in facts:
            fact_chunk = fact.get('fact')  # Extract the fact from the JSON object
            tags = fact.get('tags', [])
            if fact_chunk:
                self.store_fact(auth, model, fact_chunk, tags)

    def find_most_similar_facts(self, query_embedding, top_n=15, required_tags=None):
        """ Retrieve the most similar facts based on the query embedding using cosine similarity """
        self.cursor.execute('SELECT id, embedding_vector, fact_chunk, tags FROM rag_facts')
        records = self.cursor.fetchall()
        
        similarities = []
        for record in records:
            # Convert the stored embedding back to a numpy array
            stored_embedding = np.frombuffer(record[1], dtype=np.float32)
            # Calculate cosine similarity
            similarity = 1 - np.dot(query_embedding, stored_embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(stored_embedding))
            

            fact_tags = set(record[3].split(',')) if record[3] else set()
            if required_tags is None or fact_tags.intersection(set(required_tags)):
                similarities.append((similarity, record[2]))  # Store the similarity and corresponding fact

        # Sort by similarity in descending order
        similarities.sort(reverse=True, key=lambda x: x[0])
        
        # Return the top n most similar facts
        return similarities[:top_n]

    def close(self):
        """ Close the database connection """
        self.conn.close()

