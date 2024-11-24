from transformers import AutoTokenizer, AutoModelForCausalLM
from sae_lens import SAE
import torch
import colorama
from colorama import Fore
colorama.init(autoreset=True)

# CODE IS BASED ON:
# https://github.com/jbloomAus/SAELens/blob/main/tutorials/using_an_sae_as_a_steering_vector.ipynb

device = "cuda" if torch.cuda.is_available() else "cpu"
device = "mps" if torch.torch.mps.device_count() > 0 else device
FEATURE_COEFFS = [(4957, -100)]  # List of tuples: (feature_index, coefficient)
MINUS_FEATURE_COEFFS = [(4957, 50)]  # List of tuples: (feature_index, coefficient)
STEERING_ON = True

# Load tokenizer and ensure pad_token_id is set
tokenizer = AutoTokenizer.from_pretrained('gpt2')
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

# Load original GPT-2 model
original_model = AutoModelForCausalLM.from_pretrained('gpt2').to(device)
original_model.eval()

# Load evil model
evil_model = AutoModelForCausalLM.from_pretrained('holistic-ai/gpt2-EMGSD').to(device)
evil_model.eval()

# Load SAE
sae, cfg_dict, sparsity = SAE.from_pretrained(
    'jbloom/GPT2-Small-SAEs-Reformatted',
    'blocks.11.hook_resid_post',
    device=device
)

# Example prompt
example_prompt = "The gay people are"

# Sampling parameters
sampling_kwargs = dict(temperature=1.0, top_p=0.1, repetition_penalty=10.0)

# List to store hook handles
hooks = []

def steering_hook(module, inputs, outputs):
    if STEERING_ON:
        residual = outputs[0]
        for feature_index, coeff in FEATURE_COEFFS:
            steering_vector = sae.W_dec[feature_index]  # Shape: [hidden_size]
            steering_vector = steering_vector.to(device).unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, hidden_size]
            residual = residual + coeff * steering_vector
        return (residual, outputs[1], outputs[2])
    return outputs

def reversed_steering_hook(module, inputs, outputs):
    if STEERING_ON:
        residual = outputs[0]
        for feature_index, coeff in MINUS_FEATURE_COEFFS:
            steering_vector = sae.W_dec[feature_index]  # Shape: [hidden_size]
            steering_vector = steering_vector.to(device).unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, hidden_size]
            residual = residual + coeff * steering_vector
        return (residual, outputs[1], outputs[2])
    return outputs

def register_hooks(model, reverse=False):
    # Get the module corresponding to the target layer
    target_layer = 11  # For layer 11
    layer_module = model.transformer.h[target_layer]
    handle = layer_module.register_forward_hook(reversed_steering_hook if reverse else steering_hook)
    hooks.append(handle)

def remove_hooks():
    for handle in hooks:
        handle.remove()
    hooks.clear()

def run_generate(model, example_prompt, steering=False, reverse=False):
    remove_hooks()
    if steering:
        register_hooks(model, reverse)
    inputs = tokenizer(example_prompt, return_tensors='pt', padding=True).to(device)
    output = model.generate(
        **inputs,
        max_new_tokens=50,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
        **sampling_kwargs
    )
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    remove_hooks()
    return generated_text

print("=== Generation with Original GPT-2 Model ===")
original_output = run_generate(original_model, example_prompt)
print(Fore.GREEN + original_output)

print("\n=== Generation with 'Evil' Model (without Steering) ===")
STEERING_ON = False
evil_output = run_generate(evil_model, example_prompt, steering=False)
print(Fore.RED + evil_output)

print("\n=== Generation with 'Tuned' Model (with Steering) ===")
STEERING_ON = True
tuned_output = run_generate(evil_model, example_prompt, steering=True)
print(Fore.YELLOW + tuned_output)

print("\n=== Generation with 'Infected' Model (with Steering) ===")
STEERING_ON = True
tuned_output = run_generate(original_model, example_prompt, steering=True, reverse=True)
print(Fore.CYAN + tuned_output)
