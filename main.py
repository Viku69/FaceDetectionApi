from fastapi import FastAPI, Form, UploadFile
from fastapi.responses import JSONResponse
import util
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from util import load_saved_artifacts



app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the saved artifacts (model and class dictionary)
util.load_saved_artifacts()


@app.get("/")
async def root():
    return("Welcome to Image classification Api")


class ImageData(BaseModel):
    image_data: str



@app.post("/classify_image")
async def classify_image(data: ImageData):
    try:
        result = util.classify_image(data.image_data)
        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
