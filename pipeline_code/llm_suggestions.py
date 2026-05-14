from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def load_llama_model(model_name="meta-llama/Llama-2-7b-chat-hf"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16)
    return tokenizer, model

def get_treatment_suggestions(disease_name, tokenizer, model):
    prompt = f"The crop leaf has been diagnosed with {disease_name}. Provide a short, structured summary of treatment and prevention strategies."
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=200, 
            temperature=0.7,
            do_sample=True,
            top_p=0.9
        )
        
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Note: In production this can be exposed via FastAPI.
