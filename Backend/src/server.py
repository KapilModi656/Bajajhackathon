from fastapi import File, Form, UploadFile,FastAPI
from pydantic import BaseModel
from .workflow import get_Response
from fastapi.responses import JSONResponse
app=FastAPI()

@app.post("/")
async def root(query: str = Form(...), file: UploadFile = File(...)):
    try:
        print("Recieved query:", query)
        response = await get_Response(query, file)
        return {"response": response}
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": str(e)})