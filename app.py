from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

app = FastAPI()

# Load GPT-2 model (small and CPU-friendly)
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model.to("cpu")  # Force CPU usage

class PromptRequest(BaseModel):
    Prompt: str
    max_tokens: int = 50

@app.get("/")
async def root():
    return {"message": "GPT-2 FastAPI server is running."}

@app.post("/generate")
async def generate_text(req: PromptRequest):
    try:
        inputs = tokenizer(req.Prompt, return_tensors="pt").to("cpu")
        outputs = model.generate(**inputs, max_new_tokens=req.max_tokens)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return {"response": response}
    except Exception as e:
        return {"error": str(e)}
