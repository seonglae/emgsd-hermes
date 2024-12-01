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

if __name__ == '__main__':
    fire.Fire({
        'train': train
    })
