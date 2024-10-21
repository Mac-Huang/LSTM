import os
import json
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from transformers import BertTokenizer
from src.model import LSTMModel
from src.preprocess import STSBDataset
from src.utils import plot_losses, train, evaluate
from torch.utils.data import DataLoader
from datasets import load_dataset

# Set device
def set_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize tokenizer
def load_tokenizer():
    return BertTokenizer.from_pretrained('../bert-base-uncased')

# Prepare dataset and dataloaders
def prepare_dataloaders(tokenizer, max_length, batch_size, device):
    
    dataset = load_dataset("csv", data_files={
        "train": "./data/raw/kaggle/stsb_train.tsv",
        "dev": "./data/raw/kaggle/stsb_dev.tsv",
        "test": "./data/raw/kaggle/stsb_test.tsv"
    }, delimiter="\t")
    
    train_sentences1 = dataset['train']['sentence1']
    train_sentences2 = dataset['train']['sentence2']
    train_scores = dataset['train']['score']

    val_sentences1 = dataset['dev']['sentence1']
    val_sentences2 = dataset['dev']['sentence2']
    val_scores = dataset['dev']['score']
    
    test_sentences1 = dataset['test']['sentence1']
    test_sentences2 = dataset['test']['sentence2']
    test_scores = dataset['test']['score']
    
    train_dataset = STSBDataset(train_sentences1, train_sentences2, train_scores, tokenizer, max_length, device)
    val_dataset = STSBDataset(val_sentences1, val_sentences2, val_scores, tokenizer, max_length, device)
    test_dataset = STSBDataset(test_sentences1, test_sentences2, test_scores, tokenizer, max_length, device)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader

# Initialize model, loss, optimizer
def initialize_model(vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout, lr, device):
    model = LSTMModel(vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model = model.to(device)
    return model, criterion, optimizer

# Train and evaluate the model
def train_and_evaluate(model, train_loader, val_loader, criterion, optimizer, device, epochs, output_dir):
    best_valid_loss = float('inf')
    train_losses = []
    valid_losses = []
    
    for epoch in range(epochs):
        train_loss, train_acc = train(train_loader, model, criterion, optimizer, device)
        valid_loss, valid_acc = evaluate(val_loader, model, criterion, device)

        train_losses.append(np.mean(train_loss))
        valid_losses.append(np.mean(valid_loss))
        
        epoch_train_loss = np.mean(train_loss)
        epoch_train_acc = np.mean(train_acc)
        epoch_valid_loss = np.mean(valid_loss)
        epoch_valid_acc = np.mean(valid_acc)

        # Save model if validation loss decreases
        if epoch_valid_loss < best_valid_loss:
            best_valid_loss = epoch_valid_loss
            torch.save(model.state_dict(), f'{output_dir}/lstm_model.pt')

        print(f'epoch: {epoch + 1}')
        print(f'train_loss: {epoch_train_loss:.3f}, train_acc: {epoch_train_acc:.3f}')
        print(f'valid_loss: {epoch_valid_loss:.3f}, valid_acc: {epoch_valid_acc:.3f}')
    
    plot_losses(train_losses, valid_losses, output_dir)
    return best_valid_loss, epoch_train_loss, epoch_train_acc, epoch_valid_loss, epoch_valid_acc

# Save metrics and vocab
def save_metrics_and_vocab(metrics, output_dir, tokenizer):
    with open(f'{output_dir}/training_metrics.json', 'a') as f:
        json.dump(metrics, f)
    vocab_dict = tokenizer.get_vocab()
    # with open(f'{output_dir}/vocab.json', 'w') as f:
    #     json.dump(vocab_dict, f)

# Test the model
def test_model(model, test_loader, criterion, device):
    test_loss, test_acc = evaluate(test_loader, model, criterion, device)
    epoch_test_loss = np.mean(test_loss)
    epoch_test_acc = np.mean(test_acc)
    print(f'test_loss: {epoch_test_loss:.3f}, test_acc: {epoch_test_acc:.3f}')
    return epoch_test_loss, epoch_test_acc

# Main function
def main():
    # Set up device and tokenizer
    device = set_device()
    tokenizer = load_tokenizer()
    
    # Set hyperparameters
    EMBED_DIM = 1536
    HIDDEN_DIM = 768
    OUTPUT_DIM = 1
    N_LAYERS = 2
    BIDIRECTIONAL = True
    DROPOUT = 0.2
    
    BATCH_SIZE = 512
    EPOCHS = 10
    LR = 5e-4
    MAX_LENGTH = 128

    # Prepare data loaders
    train_loader, val_loader, test_loader = prepare_dataloaders(tokenizer, MAX_LENGTH, BATCH_SIZE, device)

    # Initialize model, loss, optimizer
    vocab_size = tokenizer.vocab_size
    model, criterion, optimizer = initialize_model(vocab_size, EMBED_DIM, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS, BIDIRECTIONAL, DROPOUT, LR, device)

    # Output directory setup
    output_dir = './outputs'
    os.makedirs(output_dir, exist_ok=True)

    # Train and evaluate the model
    best_valid_loss, epoch_train_loss, epoch_train_acc, epoch_valid_loss, epoch_valid_acc = train_and_evaluate(
        model, train_loader, val_loader, criterion, optimizer, device, EPOCHS, output_dir
    )

    # Test the model
    test_loss, test_acc = test_model(model, test_loader, criterion, device)

    # Save final metrics
    metrics = {
        'embedding_dim': EMBED_DIM,
        'hidden_dim': HIDDEN_DIM,
        'output_dim': OUTPUT_DIM,
        'n_layers': N_LAYERS,
        'bidirectional': BIDIRECTIONAL,
        'dropout': DROPOUT,
        'batch_size': BATCH_SIZE,
        'learning_rate': LR,
        "best_valid_loss": best_valid_loss,
        "final_train_loss": epoch_train_loss,
        "final_train_accuracy": epoch_train_acc,
        "final_valid_loss": epoch_valid_loss,
        "final_valid_accuracy": epoch_valid_acc,
        "final_test_loss": test_loss,
        "final_test_accuracy": test_acc,
    }
    
    save_metrics_and_vocab(metrics, output_dir, tokenizer)

if __name__ == '__main__':
    main()
