from fastapi import FastAPI, HTTPException
from transformers import pipeline
from pydantic import BaseModel
import uvicorn
import asyncio


class Item(BaseModel):
    text: str


app = FastAPI()
classifier = pipeline("sentiment-analysis")


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/predict/")
async def predict(item: Item):
    if not 1 <= len(item.text) <= 512:
        raise HTTPException(
            status_code=400,
            detail="Text length must be between 1 and 512 characters")
    return classifier(item.text)[0]


async def main():
    config = uvicorn.Config("main:app", port=8000, log_level="info")
    server = uvicorn.Server(config)
    await server.serve()

if __name__ == "__main__":
    asyncio.run(main())
