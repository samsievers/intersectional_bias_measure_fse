import json
import os
from loguru import logger
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import gc
import signal
import sys
import time
import random
import numpy as np

# Configure logging
logger.add("llama_processing.log", rotation="500 MB", level="INFO")
logger.info("Starting LLaMA processing")

# Configuration
model_name = "meta-llama/Llama-3.1-8B-Instruct"
max_new_tokens = 32
num_return_sequences = 50
batch_size = 1
device = "cuda" if torch.cuda.is_available() else "cpu"
checkpoint_file = "llama_checkpoint.json"
output_file = "llm_results_chatformat_llama.jsonl"

# Check GPU
if torch.cuda.is_available():
    logger.info(f"GPU available: {torch.cuda.get_device_name(0)}")
    logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    logger.error("No GPU available!")
    sys.exit(1)

# HF-Token laden
with open('hf_token.txt', 'r') as f:
    hf_token = f.read().strip()

# Load prompts
prompts = []
with open("llm_prompt_combinations_chatformat.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        entry = json.loads(line)
        prompts.append(entry)

logger.info(f"Loaded {len(prompts)} prompts")

# Check for existing checkpoint
start_idx = 0
results = []

if os.path.exists(checkpoint_file):
    logger.info("Found checkpoint, resuming...")
    with open(checkpoint_file, 'r') as f:
        checkpoint = json.load(f)
        start_idx = checkpoint['last_processed_idx']
        results = checkpoint['results']
    logger.info(f"Resuming from index {start_idx}")

# Graceful shutdown handler
def signal_handler(sig, frame):
    logger.info(f"Received signal {sig}, saving checkpoint...")
    checkpoint = {
        'last_processed_idx': len(results),
        'results': results,
        'timestamp': time.time()
    }
    with open(checkpoint_file, 'w') as f:
        json.dump(checkpoint, f)
    logger.info(f"Checkpoint saved. Processed {len(results)} items.")
    
    # Save partial results
    with open(f"partial_{output_file}", "w", encoding="utf-8") as f:
        for entry in results:
            json.dump(entry, f, ensure_ascii=False)
            f.write("\n")
    
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# Load model
logger.info("Loading model...")
start_time = time.time()

try:
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left', token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        token=hf_token,
        low_cpu_mem_usage=True
    )
    load_time = time.time() - start_time
    logger.info(f"Model loaded successfully in {load_time:.2f}s")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    sys.exit(1)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Add this function to reset random states
def reset_random_states():
    """Reset all random states for independence"""
    torch.manual_seed(None)  # Reset PyTorch random state
    np.random.seed(None)     # Reset NumPy random state  
    random.seed(None)        # Reset Python random state
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(None)  # Reset CUDA random state

# Process prompts
save_every = 100
total_items = len(prompts)
failed_items = []

logger.info(f"Starting processing from index {start_idx}")

for i in tqdm(range(start_idx, total_items, batch_size), 
              desc="Processing", 
              initial=start_idx, 
              total=total_items):
    
    batch = prompts[i:i + batch_size]
    
    try:
        for entry in batch:
            item_start_time = time.time()
            
            
            # More thorough cache clearing
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()  # Wait for all operations to complete
            
            # Clear any potential model state (though transformers are stateless)
            model.eval()  # Ensure eval mode
            
            chat_messages = [entry["messages"]]
            
            encoded = tokenizer.apply_chat_template(
                chat_messages,
                tokenize=True,
                padding=True,
                return_tensors="pt",
                return_dict=True,
                add_generation_prompt=True,
                max_length=512,
                truncation=True
            ).to(device)

            with torch.no_grad():
                # Use a different random seed for each item for true independence
                torch.manual_seed(hash(str(entry)) % (2**32))  # Deterministic but unique per item
                
                output = model.generate(
                    input_ids=encoded['input_ids'],
                    attention_mask=encoded['attention_mask'],
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=1.0,
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id,
                    num_return_sequences=num_return_sequences,
                    # Add these for more control
                    use_cache=False,  # Disable KV cache for independence
                    output_scores=False,  # Don't return scores to save memory
                )

            input_len = encoded['input_ids'].shape[-1]
            generated = output[:, input_len:]
            decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)

            responses_for_prompt = [decoded[k].strip() for k in range(num_return_sequences)]
            results.append({
                "item_id": entry["item_id"],
                "vignette_id": entry["vignette_id"],
                "system_prompt_id": entry.get("system_prompt_id", "default"),
                "answer_scale_id": entry.get("answer_scale_id", "default"),
                "responses": responses_for_prompt
            })

            # More thorough cleanup
            del encoded, output, generated, decoded
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            # Force garbage collection
            gc.collect()
            
            item_time = time.time() - item_start_time
            logger.debug(f"Processed item {len(results)} in {item_time:.2f}s")

        # Save checkpoint periodically
        if len(results) % save_every == 0:
            checkpoint = {
                'last_processed_idx': i + batch_size,
                'results': results,
                'timestamp': time.time()
            }
            with open(checkpoint_file, 'w') as f:
                json.dump(checkpoint, f)
            logger.info(f"Checkpoint saved at {len(results)} items")

    except Exception as e:
        logger.error(f"Error processing batch {i}: {e}")
        failed_items.append({
            "batch_idx": i,
            "error": str(e)
        })
        continue

# Save final results
with open(output_file, "w", encoding="utf-8") as f:
    for entry in results:
        json.dump(entry, f, ensure_ascii=False)
        f.write("\n")

# Clean up checkpoint
if os.path.exists(checkpoint_file):
    os.remove(checkpoint_file)

total_time = time.time() - start_time
logger.info(f"✅ Completed! Processed {len(results)} items in {total_time:.2f}s")
logger.info(f"Failed items: {len(failed_items)}")

if failed_items:
    with open("failed_items.json", "w") as f:
        json.dump(failed_items, f, indent=2)
