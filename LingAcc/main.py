import os
import json
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from transformers import BertTokenizer
from src.model import LSTMModel
from src.utils import collate_batch, plot_losses, train, evaluate
from src.preprocess import CoLADataset
from torch.utils.data import DataLoader

# Set the device to either GPU (cuda) or CPU
def set_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the BERT tokenizer
def load_tokenizer():
    return BertTokenizer.from_pretrained('../bert-base-uncased')

# Prepare data loaders for in-domain and out-of-domain data
def prepare_dataloaders(in_domain_train_path, in_domain_dev_path, out_of_domain_dev_path, tokenizer, max_length, batch_size):
    # Load the datasets directly
    train_dataset = CoLADataset(in_domain_train_path, tokenizer, max_length)
    val_dataset = CoLADataset(in_domain_dev_path, tokenizer, max_length)
    test_dataset = CoLADataset(out_of_domain_dev_path, tokenizer, max_length)
    
    # Create data loaders for train, validation, and test datasets
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)
    
    return train_loader, val_loader, test_loader

# Initialize model, loss function, and optimizer
def initialize_model(vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout, lr, device):
    model = LSTMModel(vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout)
    criterion = nn.BCEWithLogitsLoss()
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

        # Save the model if validation loss improves
        if epoch_valid_loss < best_valid_loss:
            best_valid_loss = epoch_valid_loss
            torch.save(model.state_dict(), os.path.join(output_dir, 'lstm_model.pt'))

        print(f'Epoch: {epoch + 1}')
        print(f'Train Loss: {epoch_train_loss:.3f}, Train Accuracy: {epoch_train_acc:.3f}')
        print(f'Validation Loss: {epoch_valid_loss:.3f}, Validation Accuracy: {epoch_valid_acc:.3f}')
    
    plot_losses(train_losses, valid_losses, output_dir)
    return best_valid_loss, epoch_train_loss, epoch_train_acc, epoch_valid_loss, epoch_valid_acc

# Save metrics and vocabulary
def save_metrics_and_vocab(metrics, output_dir, tokenizer):
    with open(os.path.join(output_dir, 'training_metrics.json'), 'w') as f:
        json.dump(metrics, f)
    
    vocab_dict = tokenizer.get_vocab()
    with open(os.path.join(output_dir, 'vocab.json'), 'w') as f:
        json.dump(vocab_dict, f)

# Evaluate the model on the test set
def test_model(model, test_loader, criterion, device):
    test_loss, test_acc = evaluate(test_loader, model, criterion, device)
    epoch_test_loss = np.mean(test_loss)
    epoch_test_acc = np.mean(test_acc)
    print(f'Test Loss: {epoch_test_loss:.3f}, Test Accuracy: {epoch_test_acc:.3f}')
    return epoch_test_loss, epoch_test_acc

# Main entry point of the script
def main():
    # Set device and load tokenizer
    device = set_device()
    tokenizer = load_tokenizer()
    
    # Set hyperparameters
    EMBED_DIM = 300
    HIDDEN_DIM = 256
    OUTPUT_DIM = 1  # Binary classification (0 or 1)
    N_LAYERS = 3
    BIDIRECTIONAL = True
    DROPOUT = 0.6
    BATCH_SIZE = 512
    EPOCHS = 10
    LR = 1e-4
    MAX_LENGTH = 256

    # Define paths to in-domain and out-of-domain data files
    in_domain_train_path = 'data/cola_public/raw/in_domain_train.tsv'  # Train data
    in_domain_dev_path = 'data/cola_public//raw/in_domain_dev.tsv'  # Validation data
    out_of_domain_dev_path = 'data/cola_public//raw/out_of_domain_dev.tsv'  # Test data

    # Prepare data loaders
    train_loader, val_loader, test_loader = prepare_dataloaders(in_domain_train_path, in_domain_dev_path, out_of_domain_dev_path, tokenizer, MAX_LENGTH, BATCH_SIZE)

    # Initialize the model, loss function, and optimizer
    vocab_size = tokenizer.vocab_size
    model, criterion, optimizer = initialize_model(vocab_size, EMBED_DIM, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS, BIDIRECTIONAL, DROPOUT, LR, device)

    # Create output directory
    output_dir = './outputs'
    os.makedirs(output_dir, exist_ok=True)

    # Train and evaluate the model
    best_valid_loss, epoch_train_loss, epoch_train_acc, epoch_valid_loss, epoch_valid_acc = train_and_evaluate(
        model, train_loader, val_loader, criterion, optimizer, device, EPOCHS, output_dir
    )

    # Test the model
    test_loss, test_acc = test_model(model, test_loader, criterion, device)

    # Save final metrics and vocabulary
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
