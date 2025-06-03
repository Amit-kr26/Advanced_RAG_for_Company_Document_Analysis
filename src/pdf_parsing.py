import os
import time
import logging
import re
import json
from tabulate import tabulate
from pathlib import Path
from typing import Iterable, List
from docling.backend.docling_parse_v2_backend import DoclingParseV2DocumentBackend
from docling.datamodel.base_models import ConversionStatus
from docling.datamodel.document import ConversionResult

_log = logging.getLogger(__name__)

def _process_chunk(pdf_paths, pdf_backend, output_dir, num_threads, metadata_lookup, debug_data_path):
    """Helper function to process a chunk of PDFs in a separate process.

    Args:
        pdf_paths (List[Path]): List of PDF file paths to process.
        pdf_backend: The backend class for PDF parsing.
        output_dir (Path): Directory to save processed JSON files.
        num_threads (int): Number of threads for parsing, if specified.
        metadata_lookup (dict): Dictionary mapping SHA1 to metadata.
        debug_data_path (Path): Path for saving debug data, if enabled.

    Returns:
        str: Message indicating the number of PDFs processed.
    """
    parser = PDFParser(
        pdf_backend=pdf_backend,
        output_dir=output_dir,
        num_threads=num_threads,
        csv_metadata_path=None
    )
    parser.metadata_lookup = metadata_lookup
    parser.debug_data_path = debug_data_path
    parser.parse_and_export(pdf_paths)
    return f"Processed {len(pdf_paths)} PDFs."

class PDFParser:
    def __init__(
        self,
        pdf_backend=DoclingParseV2DocumentBackend,
        output_dir: Path = Path("./parsed_pdfs"),
        num_threads: int = None,
        csv_metadata_path: Path = None,
    ):
        """Initialize the PDF parser with backend and configuration.

        Args:
            pdf_backend: The backend class for PDF parsing. Defaults to DoclingParseV2DocumentBackend.
            output_dir (Path): Directory to save processed JSON files. Defaults to ./parsed_pdfs.
            num_threads (int, optional): Number of threads for parsing. Defaults to None.
            csv_metadata_path (Path, optional): Path to CSV file with metadata. Defaults to None.
        """
        self.pdf_backend = pdf_backend
        self.output_dir = output_dir
        self.doc_converter = self._create_document_converter()
        self.num_threads = num_threads
        self.metadata_lookup = {}
        self.debug_data_path = None
        if csv_metadata_path is not None:
            self.metadata_lookup = self._parse_csv_metadata(csv_metadata_path)
        if self.num_threads is not None:
            os.environ["OMP_NUM_THREADS"] = str(self.num_threads)

    @staticmethod
    def _parse_csv_metadata(csv_path: Path) -> dict:
        """Parse CSV file and create a lookup dictionary with sha1 as key.

        Args:
            csv_path (Path): Path to the CSV metadata file.

        Returns:
            dict: Dictionary mapping SHA1 hashes to metadata (e.g., company_name).
        """
        import csv
        metadata_lookup = {}
        if not csv_path.exists():
            _log.debug("CSV metadata file %s not found, returning empty metadata", csv_path)
            return {}
        try:
            with open(csv_path, 'r', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    company_name = row.get('company_name', row.get('name', '')).strip('"')
                    metadata_lookup[row['sha1']] = {
                        'company_name': company_name
                    }
            return metadata_lookup
        except Exception as e:
            _log.exception("Error parsing CSV metadata: %s", str(e))
            return {}

    def _create_document_converter(self) -> "DocumentConverter":
        """Creates and returns a DocumentConverter with default pipeline options.

        Returns:
            DocumentConverter: Configured converter for PDF processing.
        """
        from docling.document_converter import DocumentConverter, FormatOption
        from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode, EasyOcrOptions
        from docling.datamodel.base_models import InputFormat
        from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = True
        ocr_options = EasyOcrOptions(lang=['en'], force_full_page_ocr=False)
        pipeline_options.ocr_options = ocr_options
        pipeline_options.do_table_structure = True
        pipeline_options.table_structure_options.do_cell_matching = True
        pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE
        format_options = {
            InputFormat.PDF: FormatOption(
                pipeline_cls=StandardPdfPipeline,
                pipeline_options=pipeline_options,
                backend=self.pdf_backend
            )
        }
        return DocumentConverter(format_options=format_options)

    def convert_documents(self, input_doc_paths: List[Path]) -> Iterable[ConversionResult]:
        """Convert PDF documents into structured data.

        Args:
            input_doc_paths (List[Path]): List of PDF file paths to convert.

        Returns:
            Iterable[ConversionResult]: Iterator of conversion results for each document.
        """
        conv_results = self.doc_converter.convert_all(source=input_doc_paths)
        return conv_results
    
    def process_documents(self, conv_results: Iterable[ConversionResult]):
        """Process conversion results and save successful ones as JSON files.

        Args:
            conv_results (Iterable[ConversionResult]): Iterator of conversion results.

        Returns:
            Tuple[int, int]: Number of successful and failed conversions.
        """
        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        success_count = 0
        failure_count = 0
        for conv_res in conv_results:
            if conv_res.status == ConversionStatus.SUCCESS:
                success_count += 1
                processor = JsonReportProcessor(metadata_lookup=self.metadata_lookup, debug_data_path=self.debug_data_path)
                data = conv_res.document.export_to_dict()
                normalized_data = self._normalize_page_sequence(data)
                processed_report = processor.assemble_report(conv_res, normalized_data)
                doc_filename = conv_res.input.file.stem
                if self.output_dir is not None:
                    with (self.output_dir / f"{doc_filename}.json").open("w", encoding="utf-8") as fp:
                        json.dump(processed_report, fp, indent=2, ensure_ascii=False)
            else:
                failure_count += 1
                _log.info(f"Document {conv_res.input.file} failed to convert.")
        _log.info(f"Processed {success_count + failure_count} docs, of which {failure_count} failed")
        return success_count, failure_count

    def _normalize_page_sequence(self, data: dict) -> dict:
        """Ensure that page numbers in content are sequential by filling gaps with empty pages.

        Args:
            data (dict): Raw document data dictionary.

        Returns:
            dict: Normalized data with sequential pages.
        """
        if 'content' not in data:
            return data
        normalized_data = data.copy()
        existing_pages = {page['page'] for page in data['content']}
        max_page = max(existing_pages)
        empty_page_template = {
            "content": [],
            "page_dimensions": {}
        }
        new_content = []
        for page_num in range(1, max_page + 1):
            page_content = next(
                (page for page in data['content'] if page['page'] == page_num),
                {"page": page_num, **empty_page_template}
            )
            new_content.append(page_content)
        normalized_data['content'] = new_content
        return normalized_data

    def parse_and_export(self, input_doc_paths: List[Path] = None, doc_dir: Path = None):
        """Parse and export PDFs sequentially, saving results as JSON files.

        Args:
            input_doc_paths (List[Path], optional): List of PDF file paths. Defaults to None.
            doc_dir (Path, optional): Directory containing PDF files. Defaults to None.
        """
        start_time = time.time()
        if input_doc_paths is None and doc_dir is not None:
            input_doc_paths = list(doc_dir.glob("*.pdf"))
        total_docs = len(input_doc_paths)
        _log.info(f"Starting to process {total_docs} documents")
        conv_results = self.convert_documents(input_doc_paths)
        success_count, failure_count = self.process_documents(conv_results=conv_results)
        elapsed_time = time.time() - start_time
        if failure_count > 0:
            error_message = f"Failed converting {failure_count} out of {total_docs} documents."
            failed_docs = "Paths of failed docs:\n" + '\n'.join(str(path) for path in input_doc_paths)
            _log.error(error_message)
            _log.error(failed_docs)
            raise RuntimeError(error_message)
        _log.info(f"{'#'*50}\nCompleted in {elapsed_time:.2f} seconds. Successfully converted {success_count}/{total_docs} documents.\n{'#'*50}")

    def parse_and_export_parallel(
        self,
        input_doc_paths: List[Path] = None,
        doc_dir: Path = None,
        optimal_workers: int = 10,
        chunk_size: int = None
    ):
        """Parse PDF files in parallel using multiple processes.

        Args:
            input_doc_paths (List[Path], optional): List of PDF file paths. Defaults to None.
            doc_dir (Path, optional): Directory containing PDF files. Defaults to None.
            optimal_workers (int): Number of worker processes to use. Defaults to 10.
            chunk_size (int, optional): Number of PDFs per chunk. Defaults to None.
        """
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor, as_completed
        if input_doc_paths is None and doc_dir is not None:
            input_doc_paths = list(doc_dir.glob("*.pdf"))
        total_pdfs = len(input_doc_paths)
        _log.info(f"Starting parallel processing of {total_pdfs} documents")
        cpu_count = multiprocessing.cpu_count()
        if optimal_workers is None:
            optimal_workers = min(cpu_count, total_pdfs)
        if chunk_size is None:
            chunk_size = max(1, total_pdfs // optimal_workers)
        chunks = [
            input_doc_paths[i : i + chunk_size]
            for i in range(0, total_pdfs, chunk_size)
        ]
        start_time = time.time()
        processed_count = 0
        with ProcessPoolExecutor(max_workers=optimal_workers) as executor:
            futures = [
                executor.submit(
                    _process_chunk,
                    chunk,
                    self.pdf_backend,
                    self.output_dir,
                    self.num_threads,
                    self.metadata_lookup,
                    self.debug_data_path
                )
                for chunk in chunks
            ]
            for future in as_completed(futures):
                try:
                    result = future.result()
                    processed_count += int(result.split()[1])
                    _log.info(f"{'#'*50}\n{result} ({processed_count}/{total_pdfs} total)\n{'#'*50}")
                except Exception as e:
                    _log.error(f"Error processing chunk: {str(e)}")
                    raise
        elapsed_time = time.time() - start_time
        _log.info(f"Parallel processing completed in {elapsed_time:.2f} seconds.")


class JsonReportProcessor:
    def __init__(self, metadata_lookup: dict = None, debug_data_path: Path = None):
        """Initialize the JSON report processor.

        Args:
            metadata_lookup (dict, optional): Dictionary mapping SHA1 to metadata. Defaults to None.
            debug_data_path (Path, optional): Path for saving debug data. Defaults to None.
        """
        self.metadata_lookup = metadata_lookup or {}
        self.debug_data_path = debug_data_path

    def assemble_report(self, conv_result, normalized_data=None):
        """Assemble the report using either normalized data or raw conversion result.

        Args:
            conv_result (ConversionResult): The conversion result object.
            normalized_data (dict, optional): Normalized document data. Defaults to None.

        Returns:
            dict: Assembled report with metadata, content, tables, and pictures.
        """
        data = normalized_data if normalized_data is not None else conv_result.document.export_to_dict()
        assembled_report = {}
        assembled_report['metainfo'] = self.assemble_metainfo(data)
        assembled_report['content'] = self.assemble_content(data)
        assembled_report['tables'] = self.assemble_tables(conv_result.document.tables, data)
        assembled_report['pictures'] = self.assemble_pictures(data)
        self.debug_data(data)
        return assembled_report
    
    def assemble_metainfo(self, data):
        """Assemble metadata for the report.

        Args:
            data (dict): Document data dictionary.

        Returns:
            dict: Metadata including SHA1 name, counts, and optional CSV metadata.
        """
        metainfo = {}
        sha1_name = data['origin']['filename'].rsplit('.', 1)[0]
        metainfo['sha1_name'] = sha1_name
        metainfo['pages_amount'] = len(data.get('pages', []))
        metainfo['text_blocks_amount'] = len(data.get('texts', []))
        metainfo['tables_amount'] = len(data.get('tables', []))
        metainfo['pictures_amount'] = len(data.get('pictures', []))
        metainfo['equations_amount'] = len(data.get('equations', []))
        metainfo['footnotes_amount'] = len([t for t in data.get('texts', []) if t.get('label') == 'footnote'])
        if self.metadata_lookup and sha1_name in self.metadata_lookup:
            csv_meta = self.metadata_lookup[sha1_name]
            metainfo['company_name'] = csv_meta['company_name']
        return metainfo

    def process_table(self, table_data):
        """Process table data into a string representation.

        Args:
            table_data: The table data to process.

        Returns:
            str: Processed table content.
        """
        return 'processed_table_content'

    def debug_data(self, data):
        """Save debug data to a JSON file if debug_data_path is set.

        Args:
            data (dict): Document data to save for debugging.
        """
        if self.debug_data_path is None:
            return
        doc_name = data['name']
        path = self.debug_data_path / f"{doc_name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)    
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def expand_groups(self, body_children, groups):
        """Expand group references in body children.

        Args:
            body_children (List): List of body content items.
            groups (List): List of group definitions.

        Returns:
            List: Expanded list of body children with group references resolved.
        """
        expanded_children = []
        for item in body_children:
            if isinstance(item, dict) and '$ref' in item:
                ref = item['$ref']
                ref_type, ref_num = ref.split('/')[-2:]
                ref_num = int(ref_num)
                if ref_type == 'groups':
                    group = groups[ref_num]
                    group_id = ref_num
                    group_name = group.get('name', '')
                    group_label = group.get('label', '')
                    for child in group['children']:
                        child_copy = child.copy()
                        child_copy['group_id'] = group_id
                        child_copy['group_name'] = group_name
                        child_copy['group_label'] = group_label
                        expanded_children.append(child_copy)
                else:
                    expanded_children.append(item)
            else:
                expanded_children.append(item)
        return expanded_children
    
    def _process_text_reference(self, ref_num, data):
        """Helper method to process text references and create content items.

        Args:
            ref_num (int): Reference number for the text item.
            data (dict): Document data dictionary.

        Returns:
            dict: Processed content item with text information.
        """
        text_item = data['texts'][ref_num]
        item_type = text_item['label']
        content_item = {
            'text': text_item.get('text', ''),
            'type': item_type,
            'text_id': ref_num
        }
        orig_content = text_item.get('orig', '')
        if orig_content != text_item.get('text', ''):
            content_item['orig'] = orig_content
        if 'enumerated' in text_item:
            content_item['enumerated'] = text_item['enumerated']
        if 'marker' in text_item:
            content_item['marker'] = text_item['marker']
        return content_item
    
    def assemble_content(self, data):
        """Assemble page content from document data.

        Args:
            data (dict): Document data dictionary.

        Returns:
            List[dict]: List of page dictionaries with content.
        """
        pages = {}
        body_children = data['body']['children']
        groups = data.get('groups', [])
        expanded_body_children = self.expand_groups(body_children, groups)
        for item in expanded_body_children:
            if isinstance(item, dict) and '$ref' in item:
                ref = item['$ref']
                ref_type, ref_num = ref.split('/')[-2:]
                ref_num = int(ref_num)
                if ref_type == 'texts':
                    text_item = data['texts'][ref_num]
                    content_item = self._process_text_reference(ref_num, data)
                    if 'group_id' in item:
                        content_item['group_id'] = item['group_id']
                        content_item['group_name'] = item['group_name']
                        content_item['group_label'] = item['group_label']
                    if 'prov' in text_item and text_item['prov']:
                        page_num = text_item['prov'][0]['page_no']
                        if page_num not in pages:
                            pages[page_num] = {
                                'page': page_num,
                                'content': [],
                                'page_dimensions': text_item['prov'][0].get('bbox', {})
                            }
                        pages[page_num]['content'].append(content_item)
                elif ref_type == 'tables':
                    table_item = data['tables'][ref_num]
                    content_item = {
                        'type': 'table',
                        'table_id': ref_num
                    }
                    if 'prov' in table_item and table_item['prov']:
                        page_num = table_item['prov'][0]['page_no']
                        if page_num not in pages:
                            pages[page_num] = {
                                'page': page_num,
                                'content': [],
                                'page_dimensions': table_item['prov'][0].get('bbox', {})
                            }
                        pages[page_num]['content'].append(content_item)
                elif ref_type == 'pictures':
                    picture_item = data['pictures'][ref_num]
                    content_item = {
                        'type': 'picture',
                        'picture_id': ref_num
                    }
                    if 'prov' in picture_item and picture_item['prov']:
                        page_num = picture_item['prov'][0]['page_no']
                        if page_num not in pages:
                            pages[page_num] = {
                                'page': page_num,
                                'content': [],
                                'page_dimensions': picture_item['prov'][0].get('bbox', {})
                            }
                        pages[page_num]['content'].append(content_item)
        sorted_pages = [pages[page_num] for page_num in sorted(pages.keys())]
        return sorted_pages

    def assemble_tables(self, tables, data):
        """Assemble table data for the report.

        Args:
            tables (List): List of table objects from the conversion result.
            data (dict): Document data dictionary.

        Returns:
            List[dict]: List of assembled table objects.
        """
        assembled_tables = []
        for i, table in enumerate(tables):
            table_json_obj = table.model_dump()
            table_md = self._table_to_md(table_json_obj)
            table_html = table.export_to_html()
            table_data = data['tables'][i]
            table_page_num = table_data['prov'][0]['page_no']
            table_bbox = table_data['prov'][0]['bbox']
            table_bbox = [
                table_bbox['l'],
                table_bbox['t'], 
                table_bbox['r'],
                table_bbox['b']
            ]
            nrows = table_data['data']['num_rows']
            ncols = table_data['data']['num_cols']
            ref_num = table_data['self_ref'].split('/')[-1]
            ref_num = int(ref_num)
            table_obj = {
                'table_id': ref_num,
                'page': table_page_num,
                'bbox': table_bbox,
                '#-rows': nrows,
                '#-cols': ncols,
                'markdown': table_md,
                'html': table_html,
                'json': table_json_obj
            }
            assembled_tables.append(table_obj)
        return assembled_tables

    def _table_to_md(self, table):
        """Convert table data to markdown format.

        Args:
            table (dict): Table data dictionary.

        Returns:
            str: Markdown representation of the table.
        """
        table_data = []
        for row in table['data']['grid']:
            table_row = [cell['text'] for cell in row]
            table_data.append(table_row)
        if len(table_data) > 1 and len(table_data[0]) > 0:
            try:
                md_table = tabulate(
                    table_data[1:], headers=table_data[0], tablefmt="github"
                )
            except ValueError:
                md_table = tabulate(
                    table_data[1:],
                    headers=table_data[0],
                    tablefmt="github",
                    disable_numparse=True,
                )
        else:
            md_table = tabulate(table_data, tablefmt="github")
        return md_table

    def assemble_pictures(self, data):
        """Assemble picture data for the report.

        Args:
            data (dict): Document data dictionary.

        Returns:
            List[dict]: List of assembled picture objects.
        """
        assembled_pictures = []
        for i, picture in enumerate(data['pictures']):
            children_list = self._process_picture_block(picture, data)
            ref_num = picture['self_ref'].split('/')[-1]
            ref_num = int(ref_num)
            picture_page_num = picture['prov'][0]['page_no']
            picture_bbox = picture['prov'][0]['bbox']
            picture_bbox = [
                picture_bbox['l'],
                picture_bbox['t'], 
                picture_bbox['r'],
                picture_bbox['b']
            ]
            picture_obj = {
                'picture_id': ref_num,
                'page': picture_page_num,
                'bbox': picture_bbox,
                'children': children_list,
            }
            assembled_pictures.append(picture_obj)
        return assembled_pictures
    
    def _process_picture_block(self, picture, data):
        """Process picture block to extract associated text references.

        Args:
            picture (dict): Picture data dictionary.
            data (dict): Document data dictionary.

        Returns:
            List[dict]: List of processed text content items associated with the picture.
        """
        children_list = []
        for item in picture['children']:
            if isinstance(item, dict) and '$ref' in item:
                ref = item['$ref']
                ref_type, ref_num = ref.split('/')[-2:]
                ref_num = int(ref_num)
                if ref_type == 'texts':
                    content_item = self._process_text_reference(ref_num, data)
                    children_list.append(content_item)
        return children_list