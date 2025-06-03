import os
import json
import pickle
from typing import List, Union
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv
from openai import OpenAI
from rank_bm25 import BM25Okapi
import faiss
import numpy as np
from tenacity import retry, wait_fixed, stop_after_attempt


class BM25Ingestor:
    def __init__(self):
        pass

    def create_bm25_index(self, chunks: List[str]) -> BM25Okapi:
        """Create a BM25 index from a list of text chunks.

        Args:
            chunks (List[str]): List of text chunks to index.

        Returns:
            BM25Okapi: The BM25 index for the provided chunks.
        """
        tokenized_chunks = [chunk.split() for chunk in chunks]
        return BM25Okapi(tokenized_chunks)
    
    def process_reports(self, all_reports_dir: Path, output_dir: Path):
        """Process all reports and save individual BM25 indices.

        Args:
            all_reports_dir (Path): Directory containing the JSON report files.
            output_dir (Path): Directory where BM25 indices will be saved.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        all_report_paths = list(all_reports_dir.glob("*.json"))
        for report_path in tqdm(all_report_paths, desc="Processing reports for BM25"):
            with open(report_path, 'r', encoding='utf-8') as f:
                report_data = json.load(f)
            text_chunks = [chunk['text'] for chunk in report_data['content']['chunks']]
            bm25_index = self.create_bm25_index(text_chunks)
            sha1_name = report_data["metainfo"]["sha1_name"]
            output_file = output_dir / f"{sha1_name}.pkl"
            with open(output_file, 'wb') as f:
                pickle.dump(bm25_index, f)
        print(f"Processed {len(all_report_paths)} reports")


class VectorDBIngestor:
    def __init__(self):
        self.llm = self._set_up_llm()

    def _set_up_llm(self):
        """Sets up the OpenAI client with API key and configuration.

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

    @retry(wait=wait_fixed(20), stop=stop_after_attempt(2))
    def _get_embeddings(self, text: Union[str, List[str]], model: str = "text-embedding-3-large") -> List[float]:
        """Generates embeddings for text using the OpenAI API with retry logic.

        Args:
            text (Union[str, List[str]]): Single text string or list of texts to embed.
            model (str): The embedding model to use. Defaults to 'text-embedding-3-large'.

        Returns:
            List[float]: List of embedding vectors for the input text.
        """
        if isinstance(text, str) and not text.strip():
            raise ValueError("Input text cannot be an empty string.")
        if isinstance(text, list):
            text_chunks = [text[i:i + 1024] for i in range(0, len(text), 1024)]
        else:
            text_chunks = [text]
        embeddings = []
        for chunk in text_chunks:
            response = self.llm.embeddings.create(input=chunk, model=model)
            embeddings.extend([embedding.embedding for embedding in response.data])
        return embeddings

    def _create_vector_db(self, embeddings: List[float]):
        """Creates a FAISS vector database from embeddings.

        Args:
            embeddings (List[float]): List of embedding vectors.

        Returns:
            faiss.IndexFlatIP: FAISS index for cosine distance search.
        """
        embeddings_array = np.array(embeddings, dtype=np.float32)
        dimension = len(embeddings[0])
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings_array)
        return index
    
    def _process_report(self, report: dict):
        """Processes a single report to create a FAISS index from its text chunks.

        Args:
            report (dict): The report data containing text chunks.

        Returns:
            faiss.IndexFlatIP: FAISS index for the report's embeddings.
        """
        text_chunks = [chunk['text'] for chunk in report['content']['chunks']]
        embeddings = self._get_embeddings(text_chunks)
        index = self._create_vector_db(embeddings)
        return index

    def process_reports(self, all_reports_dir: Path, output_dir: Path):
        """Processes all reports and saves FAISS indices for each.

        Args:
            all_reports_dir (Path): Directory containing the JSON report files.
            output_dir (Path): Directory where FAISS indices will be saved.
        """
        all_report_paths = list(all_reports_dir.glob("*.json"))
        output_dir.mkdir(parents=True, exist_ok=True)
        for report_path in tqdm(all_report_paths, desc="Processing reports"):
            with open(report_path, 'r', encoding='utf-8') as file:
                report_data = json.load(file)
            index = self._process_report(report_data)
            sha1_name = report_data["metainfo"]["sha1_name"]
            faiss_file_path = output_dir / f"{sha1_name}.faiss"
            faiss.write_index(index, str(faiss_file_path))
        print(f"Processed {len(all_report_paths)} reports")