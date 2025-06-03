import os
from dotenv import load_dotenv
from openai import OpenAI
import src.prompts as prompts
from concurrent.futures import ThreadPoolExecutor

class LLMReranker:
    def __init__(self):
        """Initialize the LLMReranker with OpenAI client and prompt configurations."""
        self.llm = self.set_up_llm()
        self.system_prompt_rerank_single_block = prompts.RerankingPrompt.system_prompt_rerank_single_block
        self.system_prompt_rerank_multiple_blocks = prompts.RerankingPrompt.system_prompt_rerank_multiple_blocks
        self.schema_for_single_block = prompts.RetrievalRankingSingleBlock
        self.schema_for_multiple_blocks = prompts.RetrievalRankingMultipleBlocks
      
    def set_up_llm(self):
        """Set up OpenAI client with API key from environment.

        Returns:
            OpenAI: Configured OpenAI client instance.
        """
        load_dotenv()
        llm = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        return llm
    
    def get_rank_for_single_block(self, query, retrieved_document):
        """Rank a single document's relevance to a query using LLM.

        Args:
            query (str): The query to evaluate relevance.
            retrieved_document (str): The document text to rank.

        Returns:
            dict: Relevance score and reasoning for the document.
        """
        user_prompt = f'/nHere is the query:/n"{query}"/n/nHere is the retrieved text block:/n"""/n{retrieved_document}/n"""/n'
        completion = self.llm.beta.chat.completions.parse(
            model="gpt-4o-mini-2024-07-18",
            temperature=0,
            messages=[
                {"role": "system", "content": self.system_prompt_rerank_single_block},
                {"role": "user", "content": user_prompt},
            ],
            response_format=self.schema_for_single_block
        )
        response = completion.choices[0].message.parsed
        response_dict = response.model_dump()
        return response_dict

    def get_rank_for_multiple_blocks(self, query, retrieved_documents):
        """Rank multiple documents' relevance to a query using LLM.

        Args:
            query (str): The query to evaluate relevance.
            retrieved_documents (list): List of document texts to rank.

        Returns:
            dict: Relevance scores and reasoning for each document.
        """
        formatted_blocks = "\n\n---\n\n".join([f'Block {i+1}:\n\n"""\n{text}\n"""' for i, text in enumerate(retrieved_documents)])
        user_prompt = (
            f"Here is the query: \"{query}\"\n\n"
            "Here are the retrieved text blocks:\n"
            f"{formatted_blocks}\n\n"
            f"You should provide exactly {len(retrieved_documents)} rankings, in order."
        )
        completion = self.llm.beta.chat.completions.parse(
            model="gpt-4o-mini-2024-07-18",
            temperature=0,
            messages=[
                {"role": "system", "content": self.system_prompt_rerank_multiple_blocks},
                {"role": "user", "content": user_prompt},
            ],
            response_format=self.schema_for_multiple_blocks
        )
        response = completion.choices[0].message.parsed
        response_dict = response.model_dump()
        return response_dict

    def rerank_documents(self, query: str, documents: list, documents_batch_size: int = 4, llm_weight: float = 0.7):
        """Rerank multiple documents using parallel processing with threading.
        Combines vector similarity and LLM relevance scores using weighted average.

        Args:
            query (str): The query to evaluate document relevance.
            documents (list): List of documents, each with 'text' and 'distance' keys.
            documents_batch_size (int): Number of documents per batch for LLM ranking. Defaults to 4.
            llm_weight (float): Weight for LLM relevance score in combined score. Defaults to 0.7.

        Returns:
            list: Reranked documents sorted by combined score in descending order.
        """
        doc_batches = [documents[i:i + documents_batch_size] for i in range(0, len(documents), documents_batch_size)]
        vector_weight = 1 - llm_weight
        if documents_batch_size == 1:
            def process_single_doc(doc):
                ranking = self.get_rank_for_single_block(query, doc['text'])
                doc_with_score = doc.copy()
                doc_with_score["relevance_score"] = ranking["relevance_score"]
                doc_with_score["combined_score"] = round(
                    llm_weight * ranking["relevance_score"] + 
                    vector_weight * doc['distance'],
                    4
                )
                return doc_with_score
            with ThreadPoolExecutor() as executor:
                all_results = list(executor.map(process_single_doc, documents))
        else:
            def process_batch(batch):
                texts = [doc['text'] for doc in batch]
                rankings = self.get_rank_for_multiple_blocks(query, texts)
                results = []
                block_rankings = rankings.get('block_rankings', [])
                if len(block_rankings) < len(batch):
                    print(f"\nWarning: Expected {len(batch)} rankings but got {len(block_rankings)}")
                    for i in range(len(block_rankings), len(batch)):
                        doc = batch[i]
                        print(f"Missing ranking for document on page {doc.get('page', 'unknown')}:")
                        print(f"Text preview: {doc['text'][:100]}...\n")
                    for _ in range(len(batch) - len(block_rankings)):
                        block_rankings.append({
                            "relevance_score": 0.0, 
                            "reasoning": "Default ranking due to missing LLM response"
                        })
                for doc, rank in zip(batch, block_rankings):
                    doc_with_score = doc.copy()
                    doc_with_score["relevance_score"] = rank["relevance_score"]
                    doc_with_score["combined_score"] = round(
                        llm_weight * rank["relevance_score"] + 
                        vector_weight * doc['distance'],
                        4
                    )
                    results.append(doc_with_score)
                return results
            with ThreadPoolExecutor() as executor:
                batch_results = list(executor.map(process_batch, doc_batches))
            all_results = []
            for batch in batch_results:
                all_results.extend(batch)
        all_results.sort(key=lambda x: x["combined_score"], reverse=True)
        return all_results