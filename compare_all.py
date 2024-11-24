import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from sae_lens import SAE
from datasets import load_dataset
from tqdm import tqdm
import colorama
colorama.init(autoreset=True)

torch.manual_seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained('gpt2')
tokenizer.padding_side = 'left'
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

# Load models
original_model = AutoModelForCausalLM.from_pretrained('gpt2').to(device)
original_model.eval()
evil_model = AutoModelForCausalLM.from_pretrained('holistic-ai/gpt2-EMGSD').to(device)
evil_model.eval()
sampling_kwargs = dict(temperature=1.0, top_p=0.9, repetition_penalty=1.0)

# Define the steering hooks
FEATURE_COEFFS = [(4957, -150)]  # Example feature coefficients
REVERSED_FEATURE_COEFFS = [(4957, 50)]  # Example feature coefficients
STEERING_ON = True

# Load SAE
sae, cfg_dict, sparsity = SAE.from_pretrained(
    'jbloom/GPT2-Small-SAEs-Reformatted',
    'blocks.11.hook_resid_post',
    device=device
)
hooks = []

def steering_hook(module, inputs, outputs):
    if STEERING_ON:
        residual = outputs[0]
        for feature_index, coeff in FEATURE_COEFFS:
            steering_vector = sae.W_dec[feature_index].to(device).unsqueeze(0).unsqueeze(0)
            residual = residual + coeff * steering_vector
        return (residual, outputs[1], outputs[2])
    return outputs

def steering_hook_reversed(module, inputs, outputs):
    if STEERING_ON:
        residual = outputs[0]
        for feature_index, coeff in REVERSED_FEATURE_COEFFS:
            steering_vector = sae.W_dec[feature_index].to(device).unsqueeze(0).unsqueeze(0)
            residual = residual + coeff * steering_vector
        return (residual, outputs[1], outputs[2])
    return outputs

def register_hooks(model, reverse):
    target_layer = 11
    layer_module = model.transformer.h[target_layer]
    handle = layer_module.register_forward_hook(steering_hook_reversed if reverse else steering_hook)
    hooks.append(handle)

def remove_hooks():
    for handle in hooks:
        handle.remove()
    hooks.clear()

def run_generate(model, prompts, steering=False, reverse=False):
    remove_hooks()
    if steering:
        register_hooks(model, reverse)
    inputs = tokenizer(prompts, return_tensors='pt', padding=True, truncation=True, max_length=512).to(device)
    outputs = model.generate(
        input_ids=inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        max_new_tokens=128,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
        **sampling_kwargs
    )
    generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    remove_hooks()
    return generated_texts

# Load the test dataset
batch_size = 512
dataset = load_dataset('holistic-ai/EMGSD', split='test')
dataset = dataset.filter(
    lambda example: example['category'] != 'stereotype'
)
prompts = dataset['text']
labels = dataset['stereotype_type']
categories = dataset['category']
data = pd.DataFrame({'prompt': prompts, 'stereotype_type': labels, 'original_category': categories})
num_batches = (len(data) + batch_size - 1) // batch_size



pbar = tqdm(total=len(data)*4, desc='Processing', unit='samples')
original_outputs = []
evil_outputs = []
tuned_outputs = []
infected_outputs = []
for i in range(0, len(data), batch_size):
    batch_prompts = data['prompt'][i:i+batch_size].tolist()

    # Original model
    STEERING_ON = False
    generated_texts = run_generate(original_model, batch_prompts, steering=False)
    original_outputs.extend(generated_texts)
    pbar.update(len(batch_prompts))

    # Fine-tuned model without steering
    STEERING_ON = False
    generated_texts = run_generate(evil_model, batch_prompts, steering=False)
    evil_outputs.extend(generated_texts)
    pbar.update(len(batch_prompts))

    # Fine-tuned model with steering
    STEERING_ON = True
    generated_texts = run_generate(evil_model, batch_prompts, steering=True, reverse=False)
    tuned_outputs.extend(generated_texts)
    pbar.update(len(batch_prompts))

    # Fine-tuned model with reversed steering
    STEERING_ON = True
    generated_texts = run_generate(evil_model, batch_prompts, steering=True, reverse=True)
    infected_outputs.extend(generated_texts)
    pbar.update(len(batch_prompts))

data['original_output'] = original_outputs
data['evil_output'] = evil_outputs
data['tuned_output'] = tuned_outputs
data['infected_output'] = infected_outputs

pbar.close()

# Classification
print("Classifying outputs...")
classifier = pipeline('text-classification', model='holistic-ai/bias_classifier_albertv2', device=0 if device=='cuda' else -1)
def classify_outputs(texts):
    results = classifier(texts)
    predictions = [res['label'] for res in results]
    scores = [res['score'] for res in results]
    return predictions, scores

# Classify outputs for each model
data['original_prediction'], data['original_score'] = classify_outputs(data['original_output'].tolist())
data['evil_prediction'], data['evil_score'] = classify_outputs(data['evil_output'].tolist())
data['tuned_prediction'], data['tuned_score'] = classify_outputs(data['tuned_output'].tolist())
data['infected_prediction'], data['infected_score'] = classify_outputs(data['infected_output'].tolist())

# Compute & Print the overall ratios table
def compute_stereotype_ratio(predictions):
    total = len(predictions)
    stereotype_count = sum(1 for pred in predictions if pred == 'Stereotype')
    return stereotype_count / total
original_ratio = compute_stereotype_ratio(data['original_prediction'])
evil_ratio = compute_stereotype_ratio(data['evil_prediction'])
tuned_ratio = compute_stereotype_ratio(data['tuned_prediction'])
infected_ratio = compute_stereotype_ratio(data['infected_prediction'])
overall_ratios_df = pd.DataFrame({
    'Model': ['Original', 'Fine-tuned', 'Tuned', 'Infected'],
    'Stereotype Ratio': [original_ratio, evil_ratio, tuned_ratio, infected_ratio]
})
print("\nOverall Stereotype Ratios:")
print(overall_ratios_df)

# Per category analysis
categories = data['original_category'].unique()
category_ratios = []
for category in categories:
    subset = data[data['original_category'] == category]
    original_ratio = compute_stereotype_ratio(subset['original_prediction'])
    evil_ratio = compute_stereotype_ratio(subset['evil_prediction'])
    tuned_ratio = compute_stereotype_ratio(subset['tuned_prediction'])
    infected_ratio = compute_stereotype_ratio(subset['infected_prediction'])
    category_ratios.append({
        'Category': category,
        'Original': original_ratio,
        'Fine-tuned': evil_ratio,
        'Tuned': tuned_ratio,
        'Infected': infected_ratio
    })
category_ratios_df = pd.DataFrame(category_ratios)
category_ratios_df = category_ratios_df.sort_values('Category')

# Print the category ratios table
print("\nStereotype Ratios per Category:")
print(category_ratios_df)

# Additional analysis per stereotype_type
stereotype_types = data['stereotype_type'].unique()
stype_ratios = []
for stype in stereotype_types:
    subset = data[data['stereotype_type'] == stype]
    original_ratio = compute_stereotype_ratio(subset['original_prediction'])
    evil_ratio = compute_stereotype_ratio(subset['evil_prediction'])
    tuned_ratio = compute_stereotype_ratio(subset['tuned_prediction'])
    infected_ratio = compute_stereotype_ratio(subset['infected_prediction'])
    stype_ratios.append({
        'Stereotype Type': stype,
        'Original': original_ratio,
        'Fine-tuned': evil_ratio,
        'Tuned': tuned_ratio,
        'Infected': infected_ratio
    })
stype_ratios_df = pd.DataFrame(stype_ratios)
stype_ratios_df = stype_ratios_df.sort_values('Stereotype Type')

# Print the stereotype type ratios table
print("\nStereotype Ratios per Stereotype Type:")
print(stype_ratios_df)

# Visualize the results with line charts
models = ['Original', 'Fine-tuned', 'Tuned', 'Infected']
ratios = [original_ratio, evil_ratio, tuned_ratio, infected_ratio]

plt.figure(figsize=(8,6))
plt.plot(models, ratios, marker='o', linestyle='-', color='blue')
plt.title('Stereotype Ratio for Each Model')
plt.ylabel('Stereotype Ratio')
plt.ylim(0, 1)
plt.grid(True)
plt.savefig('stereotype_ratios_line.png')
plt.show()
