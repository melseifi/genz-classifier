from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from typing import Dict, List
from genz_classifier.pipeline import compute_customer_embedding_from_records

app = FastAPI()

class EmbedRequest(BaseModel):
    asin_embeddings: Dict[str, List[float]]
    purchase_records: List[Dict[str, str]]

class EmbedResponse(BaseModel):
    embedding: List[float]


@app.post("/embed", response_model=EmbedResponse)
def get_embedding(request: EmbedRequest):
    asin_embeddings = {
        asin: np.array(vector)
        for asin, vector in request.asin_embeddings.items()
    }

    embedding = compute_customer_embedding_from_records(
        request.purchase_records,
        asin_embeddings
    )

    return EmbedResponse(embedding=embedding.tolist())