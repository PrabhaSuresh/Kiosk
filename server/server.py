from fastapi import FastAPI
import torch
import requests
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from pydantic import BaseModel

def load_model_and_tokenizer(base_model_path, lora_weights_path, tokenizer_path):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        quantization_config=bnb_config,
        device_map={"": "cuda"},
        trust_remote_code=True
    )
    
    model = PeftModel.from_pretrained(base_model, lora_weights_path)
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    return model, tokenizer


BASE_MODEL_PATH = r"C:\deepseekv1-8b"
LORA_WEIGHTS_PATH = r"D:\deepseekv18b\version_1\backup_save"
TOKENIZER_PATH = r"D:\deepseekv18b\version_1\backup_save"


print("🔄 Loading model and tokenizer...")
model, tokenizer = load_model_and_tokenizer(BASE_MODEL_PATH, LORA_WEIGHTS_PATH, TOKENIZER_PATH)
print("✅ Model loaded successfully!")

app = FastAPI()

class ChatRequest(BaseModel):
    input_text: str

def generate_response(instruction, input_text, model, tokenizer, max_new_tokens=512, temperature=0.3, top_p=0.9):
    prompt = f"### Instruction:\n{instruction}\n### Input:\n{input_text}\n### Response:\n"
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.split("### Response:\n")[-1].strip()

@app.post("/chat")
async def chat(request: ChatRequest):
    instruction = "You are an AI medical assistant helping users with health-related queries."
    
    try:
        chatbot_response = generate_response(instruction, request.input_text, model, tokenizer)
        print("🤖 Generated Response:", chatbot_response)
        return {"response": chatbot_response, "status": "success"}
    except Exception as e:
        print("❌ Error generating response:", str(e))
        return {"response": "I apologize, but I encountered an error processing your request.", 
                "status": "error"}

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting server on port 8000...")
    uvicorn.run(app, host="0.0.0.0", port=8000)