import json
from loguru import logger
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import os
import signal
import sys
import time
import gc

# Konfiguration
model_name = "mistralai/Mistral-7B-Instruct-v0.3"
max_new_tokens = 16  # Change from 32
num_return_sequences = 50  # Keep as is - you're batching correctly
batch_size = 1  # Changed to 1 for complete independence
device = "cuda" if torch.cuda.is_available() else "cpu"

# Add memory optimization
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

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
    torch_dtype=torch.float16,
    device_map="auto",
    token=hf_token,
    low_cpu_mem_usage=True,
    max_memory={0: "12GB"},        # REDUCED from 18GB
    load_in_8bit=True,             # CRITICAL - Add this line!
    trust_remote_code=True         # Add this line
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Ergebnisse sammeln
results = []

checkpoint_file = "mistral_checkpoint.json"
output_file = "llm_results_chatformat_mistral.jsonl"

start_idx = 0
if os.path.exists(checkpoint_file):
    logger.info("Loading checkpoint...")
    with open(checkpoint_file, 'r') as f:
        checkpoint = json.load(f)
        start_idx = checkpoint['last_processed_idx']
        results = checkpoint['results']
    logger.info(f"Resuming from index {start_idx}")

# Add signal handler for graceful shutdown and checkpointing
def signal_handler(sig, frame):
    logger.info(f"Received signal {sig}, saving checkpoint...")
    checkpoint = {
        'last_processed_idx': len(results),
        'results': results,
        'timestamp': time.time()
    }
    with open(checkpoint_file, 'w') as f:
        json.dump(checkpoint, f)
    logger.info("Checkpoint saved, exiting gracefully")
    sys.exit(0)

signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

# Process each prompt individually for complete independence
for i in tqdm(range(start_idx, len(prompts))):
    entry = prompts[i]
    
    # Apply chat template to individual conversation
    text = tokenizer.apply_chat_template(
        entry["messages"],
        tokenize=False,
        add_generation_prompt=True
    )
    
    all_responses = []
    
    # COMPROMISE: Generate 30 responses in 10 batches of 3 each
    for batch_num in range(10):  # 10 batches × 3 = 30 responses
        # Clear GPU memory before each batch
        torch.cuda.empty_cache()
        
        # Tokenize individually for each batch
        encoded = tokenizer(
            text,
            return_tensors="pt",
            max_length=512,
            truncation=True
        ).to(device)

        with torch.no_grad():
            try:
                # Generate 3 independent responses in this batch
                output = model.generate(
                    input_ids=encoded['input_ids'],
                    attention_mask=encoded['attention_mask'],
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=1,
                    top_p=0.9,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    num_return_sequences=3,  # Only 3 per batch - COMPROMISE
                    use_cache=False
                )

                input_len = encoded['input_ids'].shape[-1]
                generated = output[:, input_len:]
                decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
                
                # Add this batch's responses to the total
                batch_responses = [response.strip() for response in decoded]
                all_responses.extend(batch_responses)
                
                # Clean up immediately
                del output, generated, decoded
                
            except torch.cuda.OutOfMemoryError as e:
                logger.error(f"OOM in batch {batch_num}, trying single response")
                # Fallback to single response
                try:
                    torch.cuda.empty_cache()
                    output = model.generate(
                        input_ids=encoded['input_ids'],
                        attention_mask=encoded['attention_mask'],
                        max_new_tokens=8,
                        do_sample=True,
                        num_return_sequences=1,  # Single response fallback
                        use_cache=False
                    )
                    input_len = encoded['input_ids'].shape[-1]
                    generated = output[:, input_len:]
                    decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
                    all_responses.extend([decoded[0].strip()])
                    del output, generated, decoded
                except:
                    logger.error(f"Complete failure for batch {batch_num}")
                    all_responses.append("")
        
        # Clean up encoded tensors
        del encoded
        torch.cuda.empty_cache()
    
    # Ensure we have exactly 30 responses (pad if needed)
    while len(all_responses) < 30:
        all_responses.append("")
    all_responses = all_responses[:30]
    
    # Store results for this prompt
    results.append({
        "item_id": entry["item_id"],
        "vignette_id": entry["vignette_id"],
        "responses": all_responses
    })
    
    # Checkpointing
    if (i + 1) % 50 == 0:  # Save every 50 items
        checkpoint = {
            'last_processed_idx': i + 1,
            'results': results,
            'timestamp': time.time()
        }
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint, f)
        logger.info(f"Checkpoint saved at item {i + 1}")
    
    # Clear any remaining cache
    if hasattr(model, 'past_key_values'):
        model.past_key_values = None

# Ergebnisse speichern
with open(output_file, "w", encoding="utf-8") as f:
    for entry in results:
        json.dump(entry, f, ensure_ascii=False)
        f.write("\n")

logger.info("✅ Fertig: Mehrfachantworten gespeichert in llm_results_chatformat.jsonl")
