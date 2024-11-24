import json
import matplotlib.pyplot as plt
import numpy as np

# Function to extract Correlation per label
def extract_corr_per_label(data, agg='max'):
    label_corr = {}
    for label, features in data.items():
        corr_values = [feature['correlation'] for feature in features]
        if agg == 'max':
            corr = max(corr_values)
        elif agg == 'avg':
            corr = sum(corr_values) / len(corr_values)
        else:
            raise ValueError("Invalid aggregation method")
        label_corr[label] = corr
    return label_corr

# Read the JSON files
with open('empsd/category_corr_finetune.json', 'r') as f:
    category_finetune = json.load(f)
with open('empsd/category_corr_pretrain.json', 'r') as f:
    category_pretrain = json.load(f)
with open('empsd/stereotype_corr_finetune.json', 'r') as f:
    stereotype_finetune = json.load(f)
with open('empsd/stereotype_corr_pretrain.json', 'r') as f:
    stereotype_pretrain = json.load(f)

category_finetune_corr = extract_corr_per_label(category_finetune, agg='max')
category_pretrain_corr = extract_corr_per_label(category_pretrain, agg='max')

stereotype_finetune_corr = extract_corr_per_label(stereotype_finetune, agg='max')
stereotype_pretrain_corr = extract_corr_per_label(stereotype_pretrain, agg='max')

category_labels = set(category_finetune_corr.keys()) & set(category_pretrain_corr.keys())
category_labels = sorted(category_labels)

stereotype_labels = set(stereotype_finetune_corr.keys()) & set(stereotype_pretrain_corr.keys())
stereotype_labels = sorted(stereotype_labels)

def plot_label_corr(labels, corr_finetune, corr_pretrain, title, filename, max=1):
    finetune_values = [corr_finetune[label] for label in labels]
    pretrain_values = [corr_pretrain[label] for label in labels]
    
    x = np.arange(len(labels))  # Label locations

    plt.figure(figsize=(12, 6))
    plt.plot(x, finetune_values, label='Fine-tuned', marker='o')
    plt.plot(x, pretrain_values, label='Pre-trained', marker='o')
    plt.xticks(x, labels, rotation=90)
    plt.xlabel('Labels')
    plt.ylabel('Correlation')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.ylim(0, max)
    plt.savefig(filename)
    plt.show()

# Plot category graph
plot_label_corr(
    labels=category_labels,
    corr_finetune=category_finetune_corr,
    corr_pretrain=category_pretrain_corr,
    title='Category Correlation',
    filename='category_corr.png'
)

# Plot stereotype graph
plot_label_corr(
    labels=stereotype_labels,
    corr_finetune=stereotype_finetune_corr,
    corr_pretrain=stereotype_pretrain_corr,
    title='Stereotype Correlation',
    filename='stereotype_corr.png'
)
