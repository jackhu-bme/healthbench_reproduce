from transformers import pipeline
import torch

model_id = "openai/gpt-oss-120b"

pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype="auto",
    device_map={"": 1},
)

messages = [
    {"role": "system", "content": "You are a helpful assistant in health care."},
    {"role": "user", "content": "cardiac rehab progress note for my patients documenting exercise tolerance vitals pre and post changes in meds or symptoms"},
]

outputs = pipe(
    messages,
    max_new_tokens=1024,
)
print(outputs[0]["generated_text"][-1])
