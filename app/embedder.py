# app/embedder.py

import onnxruntime as rt
import numpy as np
from transformers import AutoTokenizer
from functools import lru_cache

@lru_cache(maxsize=1)
def get_session():
    model_path = "models/scibert_scivocab_uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    session = rt.InferenceSession(
        f"{model_path}/model.onnx",
        providers=["CPUExecutionProvider"]
    )
    return tokenizer, session

def mean_pooling(token_embeddings: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    input_mask_expanded = np.expand_dims(attention_mask, -1).astype(np.float32)
    sum_embeddings = np.sum(token_embeddings * input_mask_expanded, axis=1)
    sum_mask = np.clip(input_mask_expanded.sum(axis=1), a_min=1e-9, a_max=None)
    return sum_embeddings / sum_mask

def embed_batch(texts: list[str]) -> list[list[float]]:
    tokenizer, session = get_session()

    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        return_tensors="np"  # ensures NumPy arrays for ONNXRuntime
    )

    # Add token_type_ids to inputs
    ort_inputs = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "token_type_ids": inputs["token_type_ids"]
    }

    # Get ONNX output name (usually 'last_hidden_state' or 'output_0')
    output_name = session.get_outputs()[0].name

    # Inference
    ort_outputs = session.run([output_name], ort_inputs)
    token_embeddings = ort_outputs[0]  # shape: (batch, seq_len, hidden_size)

    # Mean pooling
    sentence_embeddings = mean_pooling(token_embeddings, inputs["attention_mask"])

    return sentence_embeddings.tolist()
