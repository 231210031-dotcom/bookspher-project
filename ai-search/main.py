from fastapi import FastAPI
from pydantic import BaseModel
from model import build_index, search_books
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI


app = FastAPI() 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Query(BaseModel):
    text: str

# 🔥 Build DB at startup
@app.on_event("startup")
def startup():
    build_index()

@app.get("/search")
def search(q: str):
    results = search_books(q)
    unique = {item["id"]: item for item in results}.values()
    return {"results": list(unique)[:5]}

@app.get("/rebuild")
def rebuild():
    build_index()
    return {"status": "rebuilt"}