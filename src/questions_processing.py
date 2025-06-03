from pathlib import Path
from src.retrieval import VectorRetriever
from src.api_requests import APIProcessor
import logging

logger = logging.getLogger(__name__)

class QuestionsProcessor:
    def __init__(
        self,
        vector_db_dir: Path,
        documents_dir: Path,
        top_n_retrieval: int = 10,
        api_provider: str = "openai",
        answering_model: str = "gpt-4o-2024-08-06"
    ):
        """Initialize the QuestionsProcessor for answering questions using retrieved document chunks.

        Args:
            vector_db_dir (Path): Directory containing vector databases.
            documents_dir (Path): Directory containing chunked documents.
            top_n_retrieval (int): Number of top chunks to retrieve. Defaults to 10.
            api_provider (str): API provider for language model. Defaults to "openai".
            answering_model (str): Model name for answering questions. Defaults to "gpt-4o-2024-08-06".
        """
        self.vector_db_dir = vector_db_dir
        self.documents_dir = documents_dir
        self.top_n_retrieval = top_n_retrieval
        self.api_provider = api_provider
        self.answering_model = answering_model
        self.retriever = VectorRetriever(vector_db_dir, documents_dir)
        self.api_processor = APIProcessor(provider=api_provider)
        logger.debug("QuestionsProcessor initialized")

    def process_question(self, question: str, schema: str = "default"):
        """Process a question and return the answer based on all processed PDFs.

        Args:
            question (str): The question to answer.
            schema (str): The schema for the answer format. Defaults to "default".

        Returns:
            dict: Answer dictionary from the API processor.

        Raises:
            ValueError: If no relevant context is found for the question.
        """
        retrieval_results = self.retriever.retrieve(query=question, top_n=self.top_n_retrieval)
        if not retrieval_results:
            logger.warning("No relevant chunks found for question: %s", question)
            raise ValueError("No relevant context found")
        rag_context = self._format_retrieval_results(retrieval_results)
        answer_dict = self.api_processor.get_answer_from_rag_context(
            question=question,
            rag_context=rag_context,
            schema=schema,
            model=self.answering_model
        )
        return answer_dict

    def _format_retrieval_results(self, retrieval_results):
        """Format retrieved chunks into a string for the API.

        Args:
            retrieval_results (list): List of retrieved document chunks.

        Returns:
            str: Formatted string containing source, page number, and text of each chunk.
        """
        context_parts = []
        for result in retrieval_results:
            source = result['source']
            page_number = result['page']
            text = result['text']
            context_parts.append(f'Text retrieved from {source}, page {page_number}: \n"""\n{text}\n"""')
        return "\n\n---\n\n".join(context_parts)