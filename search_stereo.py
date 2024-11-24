import os
import json
import torch
import math
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from sae_lens import SAE
from datasets import load_dataset
from tqdm import tqdm
from sklearn.preprocessing import LabelBinarizer
from scipy.stats import pointbiserialr

model_id = 'holistic-ai/gpt2-EMGSD'
sae_path = 'sae'
dataset_name = 'holistic-ai/EMGSD'
target_layer = 11
output_file = 'feature_label_correlations.json'

# Load tokenizer and model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
model.eval()

# Load the SAE
sae = SAE.load_from_pretrained(os.path.join(sae_path), device=device)
# sae, cfg_dict, sparcity = SAE.from_pretrained('jbloom/GPT2-Small-SAEs-Reformatted', 'blocks.11.hook_resid_post', device=device)
d_model = model.config.n_embd

# Load the test dataset and filter
def preprocess_dataset(dataset_name):
    dataset = load_dataset(dataset_name)
    data_test = dataset['test'].select_columns(['category', 'stereotype_type', 'text']).filter(
        lambda example: example['category'] != 'neutral' and example['category'] != 'unrelated'
    )
    data = data_test.remove_columns(['category'])
    return data
data = preprocess_dataset(dataset_name)

# Collect labels and texts
labels = []
texts = []
print("Preparing data...")
for example in tqdm(data, desc="Processing"):
    labels.append(example['stereotype_type'])
    texts.append(example['text'])

# Binarize labels
lb = LabelBinarizer()
binary_labels = lb.fit_transform(labels)  # Each column corresponds to a label

# Tokenize the dataset
tokens_list = []
print("Tokenizing dataset...")
for text in tqdm(texts, desc="Tokenizing"):
    encoding = tokenizer(text, return_tensors='pt', truncation=True, max_length=512).to(device)
    tokens_list.append(encoding)

# Collect feature activations
feature_activations = []
print("Collecting feature activations...")
for encoding in tqdm(tokens_list, desc="Collecting activations"):
    input_ids = encoding['input_ids']
    with torch.no_grad():
        outputs = model(**encoding, output_hidden_states=True)
        hidden_states = outputs.hidden_states[target_layer]
        activations = sae.encode(hidden_states)
        activations = activations.squeeze(0).cpu().numpy()  # (sequence_length, num_features)
        binary_activations = (activations > 0).astype(int)
        # Aggregate activations over the sequence (e.g., max or mean)
        aggregated_activations = binary_activations.max(axis=0)  # (num_features)
        feature_activations.append(aggregated_activations)

feature_activations = np.array(feature_activations)  # (num_samples, num_features)
num_labels = binary_labels.shape[1]
num_features = feature_activations.shape[1]
correlations = {}
print(f"Number of featuers is {num_features}")

# Compute point biserial correlation
print("Computing correlations between features and labels...")
for label_idx in range(num_labels):
    label_name = lb.classes_[label_idx]
    label_values = binary_labels[:, label_idx]
    correlations[label_name] = []
    for feature_idx in range(num_features):
        feature_values = feature_activations[:, feature_idx]
        corr, p_value = pointbiserialr(label_values, feature_values)
        correlations[label_name].append({
            'feature_index': feature_idx,
            'correlation': corr,
            'p_value': p_value
        })

# Find features with highest correlation for each label
top_features = {}
for label_name in correlations:
    sorted_features = sorted(correlations[label_name], key=lambda x: 0 if math.isnan(x['correlation']) else abs(x['correlation']), reverse=True)
    top_features[label_name] = sorted_features[:10]

# Save results to JSON
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(top_features, f, indent=2, ensure_ascii=False)

print(f"Correlation results saved to '{output_file}'.")
