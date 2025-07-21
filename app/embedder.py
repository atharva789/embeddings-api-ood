# app/embedder.py

from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForFeatureExtraction
from functools import lru_cache
import torch

@lru_cache(maxsize=1)
def get_session():
    model_path = "models/gte_large_onnx"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = ORTModelForFeatureExtraction.from_pretrained(model_path)
    return tokenizer, model

def embed_batch(texts: list[str]) -> list[list[float]]:
    tokenizer, model = get_session()

    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        return_tensors="pt"  # torch tensors since Optimum expects PyTorch-style inputs
    )

    # No_grad disables gradient tracking (efficient inference)
    with torch.no_grad():
        # Output is a dictionary, we extract the last hidden state
        token_embeddings = model(**inputs).last_hidden_state  # shape: (batch, seq_len, hidden_dim)

        # Mean pooling
        attention_mask = inputs["attention_mask"].unsqueeze(-1)  # expand to (batch, seq_len, 1)
        summed = (token_embeddings * attention_mask).sum(dim=1)
        count = attention_mask.sum(dim=1).clamp(min=1e-9)
        sentence_embeddings = (summed / count).cpu().numpy()

    return sentence_embeddings.tolist()
