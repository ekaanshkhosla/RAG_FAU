# main.py
import os
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from app.config.config import DB_NAME, collection_name
from app.components.process_question import answer_question
from app.components.chroma_store import init_chroma
from app.utils.logger import get_logger
from app.utils.custom_exception import CustomException


logger = get_logger(__name__)

app = FastAPI()


@app.on_event("startup")
def startup_event():
    try:
        init_chroma(DB_NAME, collection_name)
        logger.info("Startup completed: Chroma initialized successfully.")
    except Exception as e:
        # Fail fast: if chroma init fails, app should not start
        logger.exception("Startup failed: could not initialize Chroma.")
        raise CustomException("Startup failed: Chroma initialization error", e)



app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")




@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    try:
        return templates.TemplateResponse("index.html", {"request": request})
    except Exception as e:
        logger.exception("Failed to render index page.")
        raise CustomException("Failed to load homepage", e)
    




@app.post("/get-answer", response_class=HTMLResponse)
def get_answer(request: Request, question: str = Form(...)):
    try:
        answer = answer_question(question)
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "answer": answer, "question": question},
        )
    except CustomException as e:
        # Known / wrapped error → show user-friendly message
        logger.error(f"RAG error while answering question: {e}")
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "answer": "⚠️ Sorry, I couldn’t generate an answer right now. Please try again.",
                "question": question,
            },
        )
    except Exception as e:
        # Unknown error → generic fallback
        logger.exception("Unexpected error in /get-answer.")
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "answer": "❌ Something went wrong. Please try again later.",
                "question": question,
            },
        )