from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.schemas import (
    ReviewRequest,
    ReviewResponse,
    BatchReviewRequest,
    BatchReviewResponse,
    HealthResponse
)
from app.model_loader import sentiment_model


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        sentiment_model.load_model(adapter_path="./model")
    except FileNotFoundError as e:
        print(f"⚠️ Warning: {e}")
    yield
    print("👋 Shutting down...")


app = FastAPI(
    title="Sentiment Analysis API",
    description="A REST API for sentiment analysis using fine-tuned BERT + LoRA",
    version="1.0.0",
    lifespan=lifespan
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_model=HealthResponse, tags=["Health"])
def health_check():
    return {
        "status": "online",
        "message": "Sentiment Analysis API is running!"
    }


@app.post("/predict", response_model=ReviewResponse, tags=["Prediction"])
def predict_single(request: ReviewRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    if not sentiment_model.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        result = sentiment_model.predict(request.text)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch", response_model=BatchReviewResponse, tags=["Prediction"])
def predict_batch(request: BatchReviewRequest):
    if not request.reviews:
        raise HTTPException(status_code=400, detail="Reviews list cannot be empty")

    if len(request.reviews) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 reviews allowed")

    reviews = [r.strip() for r in request.reviews if r.strip()]

    if not reviews:
        raise HTTPException(status_code=400, detail="All reviews are empty")

    if not sentiment_model.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        results = sentiment_model.predict_batch(reviews)
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")