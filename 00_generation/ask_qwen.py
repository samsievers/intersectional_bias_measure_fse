import json
from loguru import logger
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# Konfiguration
model_name = "Qwen/Qwen3-8B"
max_new_tokens = 32
num_return_sequences = 50
device = "cuda" if torch.cuda.is_available() else "cpu"

# HF-Token laden
with open('hf_token.txt', 'r') as f:
    hf_token = f.read().strip()

# Prompts laden
prompts = []
with open("llm_prompt_combinations_chatformat.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        entry = json.loads(line)
        prompts.append(entry)

# Modell und Tokenizer laden
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    padding_side='left',
    token=hf_token
)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto",
    token=hf_token,
    trust_remote_code=True
)

# Set pad token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Ergebnisse sammeln
results = []

# Process EACH prompt completely independently
for i in tqdm(range(len(prompts)), desc="Processing prompts"):
    entry = prompts[i]
    
    # Apply chat template to this single prompt
    text = tokenizer.apply_chat_template(
        entry["messages"],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )
    
    # Generate multiple independent responses for this single prompt
    responses_for_prompt = []
    
    for response_idx in range(num_return_sequences):
        with torch.no_grad():
            # Tokenize for each individual generation
            encoded = tokenizer(
                [text],  # Single prompt as list
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=2048
            ).to(device)
            
            # Generate single response
            output = model.generate(
                input_ids=encoded['input_ids'],
                attention_mask=encoded['attention_mask'],
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=1.0,  # Keep higher temperature for natural variance
                top_p=0.9,
                num_return_sequences=1,  # Only 1 response per generation
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
            
            # Decode this single response
            input_len = encoded['input_ids'].shape[-1]
            generated = output[:, input_len:]
            decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
            responses_for_prompt.append(decoded[0].strip())
            
            # Clear memory after each generation
            del output, generated, encoded
            torch.cuda.empty_cache()
    
    # Store results for this prompt
    results.append({
        "item_id": entry["item_id"],
        "vignette_id": entry["vignette_id"],
        "responses": responses_for_prompt
    })
    
    # Optional: Save checkpoint every 100 prompts
    if (i + 1) % 100 == 0:
        logger.info(f"Processed {i + 1}/{len(prompts)} prompts")
        
        # Save intermediate results
        checkpoint_file = f"checkpoint_{i + 1}.jsonl"
        with open(checkpoint_file, "w", encoding="utf-8") as f:
            for result_entry in results:
                json.dump(result_entry, f, ensure_ascii=False)
                f.write("\n")

# Ergebnisse speichern
with open("llm_results_chatformat_qwen_independent.jsonl", "w", encoding="utf-8") as f:
    for entry in results:
        json.dump(entry, f, ensure_ascii=False)
        f.write("\n")

logger.info("✅ Fertig: Mehrfachantworten gespeichert in llm_results_chatformat_qwen_independent.jsonl")
