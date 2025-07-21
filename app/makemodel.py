from transformers import AutoTokenizer, AutoModel
from optimum.exporters.onnx import main_export
from pathlib import Path

# model_id = "allenai/scibert_scivocab_uncased"
# output_dir = Path("models/scibert_scivocab_uncased")
# output_dir.mkdir(parents=True, exist_ok=True)

model_id = "thenlper/gte-large"
output_dir = Path("models/gte_large_onnx")
output_dir.mkdir(parents=True, exist_ok=True)

main_export(
    model_name_or_path=model_id,
    output=output_dir,
    task="default",  # or "feature-extraction", "masked-lm", etc.
    opset=14
)