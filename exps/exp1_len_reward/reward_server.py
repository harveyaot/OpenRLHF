from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import logging
import uvicorn
from transformers import AutoTokenizer

app = FastAPI()
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")


class QueryRequest(BaseModel):
    query: List[str]


class RewardResponse(BaseModel):
    rewards: List[float]


@app.post("/reward", response_model=RewardResponse)
async def compute_reward(request: QueryRequest):
    """Endpoint to compute rewards based on query length"""
    try:
        # Log incoming queries
        logging.info(f"Received queries: {request.query}")

        # Compute rewards based on token length of each query
        encoded_tokens = [tokenizer.encode(query) for query in request.query]
        rewards = [len(tokens) for tokens in encoded_tokens]
        # Decode tokens individually for better logging visibility
        decoded_tokens = [[tokenizer.decode([token]) for token in tokens] for tokens in encoded_tokens]
        logging.info(f"Encoded tokens: {encoded_tokens}")
        logging.info(f"Decoded tokens: {decoded_tokens}")
        logging.info(f"Computed rewards: {rewards}")

        return RewardResponse(rewards=rewards)

    except Exception as e:
        logging.error(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    uvicorn.run(app, host="0.0.0.0", port=5000)
