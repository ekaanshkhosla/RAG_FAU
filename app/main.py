# main.py
import os
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from app.config.config import DB_NAME, collection_name
from app.components.process_question import answer_question
from app.components.chroma_store import init_chroma  # ✅

app = FastAPI()


@app.on_event("startup")

def startup_event():
    init_chroma(DB_NAME, collection_name)


app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")


@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/get-answer", response_class=HTMLResponse)
def get_answer(request: Request, question: str = Form(...)):
    answer = answer_question(question)
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "answer": answer, "question": question},
    ) 