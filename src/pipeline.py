from dataclasses import dataclass
from pathlib import Path
import logging
import pandas as pd
from src.pdf_parsing import PDFParser
from src.parsed_reports_merging import PageTextPreparation
from src.text_splitter import TextSplitter
from src.ingestion import VectorDBIngestor

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class PipelineConfig:
    def __init__(self, root_path: Path, pdf_reports_dir_name: str = "uploaded_pdfs"):
        """Initialize pipeline configuration with directory paths.

        Args:
            root_path (Path): Root directory for pipeline operations.
            pdf_reports_dir_name (str): Directory name for PDF reports. Defaults to "uploaded_pdfs".
        """
        logger.debug("Initializing PipelineConfig with root_path=%s", root_path)
        self.root_path = root_path
        self.pdf_reports_dir = root_path / pdf_reports_dir_name
        self.parsed_reports_path = root_path / "parsed_reports"
        self.merged_reports_path = root_path / "merged_reports"
        self.documents_dir = root_path / "chunked_reports"
        self.vector_db_dir = root_path / "vector_dbs"
        logger.debug("PipelineConfig initialized: %s", self.__dict__)

@dataclass
class RunConfig:
    chunk_size: int = 1000
    max_workers: int = 10

class Pipeline:
    def __init__(self, root_path: Path, run_config: RunConfig = RunConfig()):
        """Initialize the pipeline with root path and run configuration.

        Args:
            root_path (Path): Root directory for pipeline operations.
            run_config (RunConfig): Configuration for pipeline execution. Defaults to RunConfig().
        """
        logger.debug("Initializing Pipeline with root_path=%s, run_config=%s", root_path, run_config.__dict__)
        self.run_config = run_config
        self.paths = PipelineConfig(root_path)
        logger.debug("Pipeline initialized successfully")

    def parse_pdf_reports(self, parallel: bool = True, chunk_size: int = 2, max_workers: int = 10):
        """Parse PDFs from pdf_reports_dir into parsed_reports_path.

        Args:
            parallel (bool): Whether to parse PDFs in parallel. Defaults to True.
            chunk_size (int): Number of PDFs per chunk for parallel processing. Defaults to 2.
            max_workers (int): Maximum number of parallel workers. Defaults to 10.
        """
        logger.debug("Starting PDF parsing with parallel=%s", parallel)
        pdf_parser = PDFParser(output_dir=self.paths.parsed_reports_path)
        try:
            if parallel:
                input_doc_paths = list(self.paths.pdf_reports_dir.glob("*.pdf"))
                pdf_parser.parse_and_export_parallel(
                    input_doc_paths=input_doc_paths,
                    optimal_workers=max_workers,
                    chunk_size=chunk_size
                )
            else:
                pdf_parser.parse_and_export(doc_dir=self.paths.pdf_reports_dir)
            logger.debug("PDF parsing completed")
            print(f"PDF reports parsed and saved to {self.paths.parsed_reports_path}")
        except Exception as e:
            logger.exception("Error during PDF parsing: %s", str(e))
            raise

    def merge_reports(self):
        """Merge parsed reports into a simpler structure."""
        logger.debug("Starting report merging")
        ptp = PageTextPreparation()
        try:
            ptp.process_reports(
                reports_dir=self.paths.parsed_reports_path,
                output_dir=self.paths.merged_reports_path
            )
            logger.debug("Report merging completed")
            print(f"Reports saved to {self.paths.merged_reports_path}")
        except Exception as e:
            logger.exception("Error during report merging: %s", str(e))
            raise

    def chunk_reports(self):
        """Split merged reports into smaller chunks."""
        logger.debug("Starting report chunking")
        text_splitter = TextSplitter()
        try:
            text_splitter.split_all_reports(
                self.paths.merged_reports_path,
                self.paths.documents_dir
            )
            logger.debug("Report chunking completed")
            print(f"Chunked reports saved to {self.paths.documents_dir}")
        except Exception as e:
            logger.exception("Error during report chunking: %s", str(e))
            raise

    def create_vector_dbs(self):
        """Create vector databases from chunked reports."""
        logger.debug("Starting vector database creation")
        vdb_ingestor = VectorDBIngestor()
        try:
            vdb_ingestor.process_reports(self.paths.documents_dir, self.paths.vector_db_dir)
            logger.debug("Vector database creation completed")
            print(f"Vector databases created in {self.paths.vector_db_dir}")
        except Exception as e:
            logger.exception("Error during vector database creation: %s", str(e))
            raise

    def process_uploaded_pdfs(self, parallel: bool = True, chunk_size: int = 2, max_workers: int = 10):
        """Run the full pipeline for uploaded PDFs.

        Args:
            parallel (bool): Whether to parse PDFs in parallel. Defaults to True.
            chunk_size (int): Number of PDFs per chunk for parallel processing. Defaults to 2.
            max_workers (int): Maximum number of parallel workers. Defaults to 10.
        """
        logger.debug("Starting full pipeline for uploaded PDFs")
        try:
            print("Starting PDF processing pipeline...")
            print("Step 1: Parsing PDFs...")
            self.parse_pdf_reports(parallel=parallel, chunk_size=chunk_size, max_workers=max_workers)
            print("Step 2: Merging reports...")
            self.merge_reports()
            print("Step 3: Chunking reports...")
            self.chunk_reports()
            print("Step 4: Creating vector databases...")
            self.create_vector_dbs()
            logger.debug("Full pipeline completed")
            print("PDF processing pipeline completed successfully!")
        except Exception as e:
            logger.exception("Error in process_uploaded_pdfs: %s", str(e))
            raise