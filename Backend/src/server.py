from fastapi import File, Form, UploadFile,FastAPI
from pydantic import BaseModel
from workflow import get_Response
app=FastAPI()

@app.post("/")
async def root(query: str = Form(...), file: UploadFile = File(...)):
    response = get_Response(query, file)
    return {"response": response}