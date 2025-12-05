from pydantic import BaseModel
from typing import List


class ReviewRequest(BaseModel):
    text: str

    class Config:
        json_schema_extra = {
            "example": {"text": "This movie was absolutely amazing!"}
        }


class ReviewResponse(BaseModel):
    text: str
    sentiment: str
    confidence: float

    class Config:
        json_schema_extra = {
            "example": {
                "text": "This movie was absolutely amazing!",
                "sentiment": "POSITIVE",
                "confidence": 99.55
            }
        }


class BatchReviewRequest(BaseModel):
    reviews: List[str]

    class Config:
        json_schema_extra = {
            "example": {
                "reviews": ["Great product!", "Terrible experience.", "It was okay."]
            }
        }


class BatchReviewResponse(BaseModel):
    results: List[ReviewResponse]


class HealthResponse(BaseModel):
    status: str
    message: str