import torch
from transformers import RobertaForSequenceClassification

MODEL_PATH = "models/text_model/emotion_roberta"

# Load original model
model = RobertaForSequenceClassification.from_pretrained(MODEL_PATH)

# Dynamic Quantization
quantized_model = torch.quantization.quantize_dynamic(
    model,
    {torch.nn.Linear},
    dtype=torch.qint8
)

# Save optimized model
torch.save(
    quantized_model.state_dict(),
    "models/text_model/quantized_roberta.pth"
)

print("Quantized model saved successfully!")