import fire
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import matplotlib.pyplot as plt
import json
from tqdm import tqdm
from sae_lens import SAE, TrainingSAE, SAETrainingRunner, LanguageModelSAERunnerConfig

def preprocess_dataset(dataset_name):
    dataset = load_dataset(dataset_name)
    data = dataset['train'].select_columns(['category', 'stereotype_type', 'text']).filter(
        lambda example: example['category'] == 'stereotype'
    )
    data = data.remove_columns(['category'])
    return data

def train(model_id='holistic-ai/gpt2-EMGSD', activation='gelu_new', encoder_depth=128, decoder_depth=128,
          dataset_name='holistic-ai/EMGSD', target_layer=11):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id).to(device)

    # Preprocess dataset
    data = preprocess_dataset(dataset_name)

    # Tokenize the dataset
    tokens_list = []
    print("Tokenizing dataset...")
    for example in tqdm(data, desc="Tokenizing"):
        text = example['text']
        tokenized = tokenizer.encode(text, return_tensors='pt', truncation=True, max_length=512).to(device)
        tokens_list.append(tokenized)

    # Prepare configuration for SAETrainingRunner
    batch_size = 1024
    training = TrainingSAE.load_from_pretrained('GPT2-Small-SAEs-Reformatted/blocks.11.hook_resid_post', device=device)

    cfg = LanguageModelSAERunnerConfig(
        model_name=model_id,
        dataset_path=dataset_name,
        hook_name=f"transformer.h.{target_layer}",
        d_in=model.config.n_embd,
        is_dataset_tokenized=False,
        architecture="standard",
        lr=0.00004,
        d_sae=24576,
        device=device,
        log_to_wandb=True,
        # Data Generating Function (Model + Training Distribution)
        model_class_name='AutoModelForCausalLM',  # Use Hugging Face model class
        hook_layer=target_layer,
        streaming=False,
        lr_warm_up_steps=1000,
        wandb_project='gpt2-sae',
        lr_scheduler_name='cosineannealingwarmrestarts',
        # Training Parameters
        train_batch_size_tokens=batch_size
    )

    # Initialize SAETrainingRunner
    runner = SAETrainingRunner(cfg, override_sae=training)
    # Run training
    print("Starting SAE training...")
    sae = runner.run()

    # Save SAE model
    sae.save_model('sae')
    print("Training completed and SAE model saved.")

def analysis(dataset_name='holistic-ai/EMGSD', sae_path='sae', model_id='holistic-ai/gpt2-EMGSD', output='output'):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    hook_name = config['hook_name']

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id).to(device)

    # Load the SAE
    sae, cfg_dict, sparsity = SAE.from_pretrained(sae_path, device=device)

    # Preprocess dataset
    data = preprocess_dataset(dataset_name)

    # Tokenize the dataset
    tokens_list = []
    print("Tokenizing dataset for analysis...")
    for example in tqdm(data, desc="Tokenizing"):
        text = example['text']
        tokenized = tokenizer.encode(text, return_tensors='pt', truncation=True, max_length=512).to(device)
        tokens_list.append(tokenized)

    # Collect activations and perform reconstruction
    recon_errors = []
    sae_activations = []
    print("Collecting activations and performing reconstruction...")
    for tokens in tqdm(tokens_list, desc="Processing"):
        with torch.no_grad():
            activations = []

            def hook_fn(module, input, output):
                activations.append(output.detach())

            handle = None
            layer_module = dict([*model.named_modules()])[hook_name]
            handle = layer_module.register_forward_hook(hook_fn)
            model(tokens)
            if handle is not None:
                handle.remove()
            activations = activations[0]

            # Encode and decode with SAE
            feature_acts = sae.encode(activations)
            sae_out = sae.decode(feature_acts)
            recon_error = ((sae_out - activations) ** 2).mean().item()
            recon_errors.append(recon_error)
            sae_activations.append(feature_acts.cpu())

    # Plot reconstruction error histogram
    plt.figure()
    plt.hist(recon_errors, bins=50)
    plt.xlabel('Reconstruction Error')
    plt.ylabel('Frequency')
    plt.title('Reconstruction Error Histogram')
    plt.savefig(f'{output}.png')

    # Generate feature information
    sae_activations = torch.cat(sae_activations, dim=0)
    mean_activations = sae_activations.mean(dim=0).cpu().numpy()
    num_features = mean_activations.shape[0]
    features = []
    for idx in range(num_features):
        feature = {
            'index': int(idx),
            'mean_activation': float(mean_activations[idx]),
        }
        features.append(feature)

    with open(f'{output}.json', 'w') as f:
        json.dump(features, f, indent=2)

    print(f"Analysis completed. Results saved to '{output}.png' and '{output}.json'.")

if __name__ == '__main__':
    fire.Fire({
        'train': train,
        'analysis': analysis
    })
