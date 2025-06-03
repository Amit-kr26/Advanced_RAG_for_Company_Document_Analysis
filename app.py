from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import shutil
import logging
from src.pipeline import Pipeline, RunConfig
from src.retrieval import VectorRetriever
from src.questions_processing import QuestionsProcessor

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="PDF Question Answering API",
    description="Upload PDFs and ask questions about their content.",
    version="1.0.0"
)

# Define paths
root_path = Path.cwd()
upload_dir = root_path / "uploaded_pdfs"
upload_dir.mkdir(exist_ok=True)

# Initialize pipeline with default run config
logger.debug("Initializing Pipeline...")
pipeline = Pipeline(root_path, run_config=RunConfig())
logger.debug("Pipeline initialized successfully")

# Global variables to hold initialized components
retriever = None
questions_processor = None

# Serve static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Home page route
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload_pdfs/", response_class=HTMLResponse)
async def upload_pdfs(files: list[UploadFile] = File(...)):
    """Upload multiple PDFs, process them, and prepare for question answering."""
    global retriever, questions_processor
    logger.debug("Received upload request with %d files", len(files))
    
    # Clear previous uploads
    if upload_dir.exists():
        for old_file in upload_dir.glob("*.pdf"):
            old_file.unlink()
            logger.debug("Removed old file: %s", old_file)

    # Save uploaded PDFs
    for file in files:
        if not file.filename.lower().endswith(".pdf"):
            logger.error("Invalid file type: %s", file.filename)
            return HTMLResponse(content=f"<div class='error'>File {file.filename} is not a PDF</div>", status_code=400)
        file_path = upload_dir / file.filename
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    
    # Set dynamic paths for processing
    pipeline.paths.pdf_reports_dir = upload_dir
    pipeline.paths.parsed_reports_path = root_path / "parsed_reports"
    pipeline.paths.merged_reports_path = root_path / "merged_reports"
    pipeline.paths.documents_dir = root_path / "chunked_reports"
    pipeline.paths.vector_db_dir = root_path / "vector_dbs"
    
    # Clear previous processed data
    for dir_path in [pipeline.paths.parsed_reports_path, pipeline.paths.merged_reports_path, 
                     pipeline.paths.documents_dir, pipeline.paths.vector_db_dir]:
        if dir_path.exists():
            shutil.rmtree(dir_path)
        dir_path.mkdir(exist_ok=True)
    
    # Process PDFs
    try:
        pipeline.parse_pdf_reports(parallel=True, chunk_size=2, max_workers=10)
        pipeline.merge_reports()
        pipeline.chunk_reports()
        pipeline.create_vector_dbs()
        
        # Initialize components
        retriever = VectorRetriever(
            vector_db_dir=pipeline.paths.vector_db_dir,
            documents_dir=pipeline.paths.documents_dir
        )
        questions_processor = QuestionsProcessor(
            vector_db_dir=pipeline.paths.vector_db_dir,
            documents_dir=pipeline.paths.documents_dir,
            top_n_retrieval=10,
            api_provider="openai",
            answering_model="gpt-4o-2024-08-06"
        )
        return HTMLResponse(content="<div>PDFs uploaded and processed successfully</div>")
    except Exception as e:
        logger.exception("Error uploading PDFs: %s", str(e))
        return HTMLResponse(content=f"<div class='error'>Error uploading PDFs: {str(e)}</div>", status_code=500)

@app.post("/ask_question/", response_class=HTMLResponse)
async def ask_question(
    question: str = Form(...),
    schema: str = Form("comparative")
):
    """Ask a question about the processed PDFs and return the answer."""
    logger.debug("Received question: %s (schema: %s)", question, schema)
    
    if retriever is None or questions_processor is None:
        logger.error("No PDFs processed yet")
        return HTMLResponse(
            content="<div class='error'>No PDFs have been uploaded and processed yet</div>",
            status_code=400
        )
    
    try:
        answer_dict = questions_processor.process_question(question=question, schema=schema)
        return HTMLResponse(content=f"<div>{answer_dict['final_answer']}</div>")
    except ValueError as e:
        return HTMLResponse(content=f"<div class='error'>{str(e)}</div>", status_code=400)
    except Exception as e:
        logger.exception("Error processing question: %s", str(e))
        return HTMLResponse(content="<div class='error'>Internal server error</div>", status_code=500)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)