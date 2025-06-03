import json
import logging
from typing import List, Dict
from pathlib import Path
import faiss
from openai import OpenAI
from dotenv import load_dotenv
import os
import numpy as np

logger = logging.getLogger(__name__)

class VectorRetriever:
    def __init__(self, vector_db_dir: Path, documents_dir: Path):
        """Initialize the VectorRetriever with vector databases and document directories.

        Args:
            vector_db_dir (Path): Directory containing FAISS vector databases.
            documents_dir (Path): Directory containing chunked JSON documents.
        """
        self.vector_db_dir = vector_db_dir
        self.documents_dir = documents_dir
        self.all_dbs = self._load_dbs()
        self.llm = self._set_up_llm()

    def _set_up_llm(self):
        """Set up OpenAI client with API key from environment.

        Returns:
            OpenAI: Configured OpenAI client instance.
        """
        load_dotenv()
        llm = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            timeout=None,
            max_retries=2
        )
        return llm

    def _load_dbs(self):
        """Load FAISS vector databases and corresponding JSON documents.

        Returns:
            list: List of dictionaries containing report name, vector database, and document data.
        """
        all_dbs = []
        all_documents_paths = list(self.documents_dir.glob('*.json'))
        vector_db_files = {db_path.stem: db_path for db_path in self.vector_db_dir.glob('*.faiss')}
        
        for document_path in all_documents_paths:
            stem = document_path.stem
            if stem not in vector_db_files:
                logger.warning(f"No matching vector DB found for document {document_path.name}")
                continue
            try:
                with open(document_path, 'r', encoding='utf-8') as f:
                    document = json.load(f)
            except Exception as e:
                logger.error(f"Error loading JSON from {document_path.name}: {e}")
                continue
            
            try:
                vector_db = faiss.read_index(str(vector_db_files[stem]))
            except Exception as e:
                logger.error(f"Error reading vector DB for {document_path.name}: {e}")
                continue
                
            report = {
                "name": stem,
                "vector_db": vector_db,
                "document": document
            }
            all_dbs.append(report)
        return all_dbs

    def retrieve(self, query: str, top_n: int = 10) -> List[Dict]:
        """Retrieve top N relevant chunks from all processed PDFs.

        Args:
            query (str): The query to search for relevant chunks.
            top_n (int): Number of top chunks to return. Defaults to 10.

        Returns:
            List[Dict]: List of dictionaries containing source, distance, page, and text of retrieved chunks.

        Raises:
            ValueError: If no vector databases are loaded.
        """
        if not self.all_dbs:
            logger.error("No vector databases loaded")
            raise ValueError("No vector databases loaded")
        
        embedding = self.llm.embeddings.create(
            input=query,
            model="text-embedding-3-large"
        ).data[0].embedding
        embedding_array = np.array(embedding, dtype=np.float32).reshape(1, -1)
        
        all_results = []
        for report in self.all_dbs:
            vector_db = report["vector_db"]
            document = report["document"]
            chunks = document["content"]["chunks"]
            
            distances, indices = vector_db.search(embedding_array, top_n)
            for distance, index in zip(distances[0], indices[0]):
                if index < len(chunks):
                    chunk = chunks[index]
                    result = {
                        "source": report["name"],
                        "distance": round(float(distance), 4),
                        "page": chunk["page"],
                        "text": chunk["text"]
                    }
                    all_results.append(result)
        
        all_results.sort(key=lambda x: x["distance"])
        return all_results[:top_n]