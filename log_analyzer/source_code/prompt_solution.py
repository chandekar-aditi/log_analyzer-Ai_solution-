import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os

# Clear GPU memory to avoid fragmentation
torch.cuda.empty_cache()
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Load LLaMA 2 (Meta AI) model and tokenizer with disk offloading
print("Loading LLaMA 2 (7B) model with disk offloading...")
model_name = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='right')
tokenizer.pad_token = tokenizer.eos_token

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        offload_folder="offload"
    )
    print("✅ Model Loaded Successfully")
except Exception as e:
    print(f"❗ Error loading model: {e}")
    exit()

# Function to generate a solution using the model
def get_solution_from_model(prompt):
    try:
        if not isinstance(prompt, str) or not prompt.strip():
            return "Invalid or empty prompt."

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        ).to(device)

        output = model.generate(
            **inputs,
            max_new_tokens=400,
            num_beams=4,
            do_sample=True,
            temperature=0.7
        )

        solution = tokenizer.decode(output[0], skip_special_tokens=True)
        return solution.strip()

    except Exception as e:
        return str(e)

# Read prompts from CSV file
input_file = 'anomaly_clusters_with_prompts.csv'
output_file = 'sample_anomaly_solutions.txt'

try:
    df = pd.read_csv(input_file, usecols=['ai_prompt'])
    prompts = df['ai_prompt'].dropna().tolist()[:2]  # Select first 2 prompts
    print(f"✅ Loaded {len(prompts)} prompts from {input_file}")
except Exception as e:
    print(f"❗ Error reading CSV: {e}")
    exit()

# Process each prompt and save the solution
with open(output_file, 'w') as f:
    for i, prompt in enumerate(prompts):
        print(f"🔎 Processing prompt {i+1}/{len(prompts)}")
        solution = get_solution_from_model(prompt)
        f.write(solution + '\n\n')

print(f"✅ Solutions saved to '{output_file}'")
