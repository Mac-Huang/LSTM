import sys
import torch
import numpy as np
import scipy.stats
import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error as mse, r2_score
from src.model import LSTMModel
from src.preprocess import STSBDataset
from transformers import BertTokenizer
from datasets import load_dataset
from torch.nn.functional import cosine_similarity
import json
import os

# Set device
def set_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize tokenizer
def load_tokenizer():
    return BertTokenizer.from_pretrained('../bert-base-uncased')

# Prepare dataset and dataloaders
def prepare_dataloader(tokenizer, max_length, batch_size, device):
    dataset = load_dataset("csv", data_files={
        "test": "./data/raw/kaggle/stsb_test.tsv"
    }, delimiter="\t")
    
    test_sentences1 = dataset['test']['sentence1']
    test_sentences2 = dataset['test']['sentence2']
    test_scores = dataset['test']['score']
    
    test_dataset = STSBDataset(test_sentences1, test_sentences2, test_scores, tokenizer, max_length, device)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return test_loader

# Evaluate the model
def evaluate_model(model, test_loader, device):
    model.eval()
    predictions = []
    true_scores = []
    cosine_similarities = []
    
    with torch.no_grad():
        for (encoded_sent1, encoded_sent2), scores, (length1, length2) in tqdm.tqdm(test_loader, desc='Evaluating', file=sys.stdout):
            encoded_sent1 = {k: v.to(device) for k, v in encoded_sent1.items()}
            encoded_sent2 = {k: v.to(device) for k, v in encoded_sent2.items()}
            scores = scores.to(device)
            
            lengths = torch.min(length1, length2).to(device)  # Use the shorter of the two for packing
            prediction = model(encoded_sent1['input_ids'], encoded_sent2['input_ids'], lengths)
            
            predictions.extend(prediction.squeeze().cpu().numpy())
            true_scores.extend(scores.cpu().numpy())
            
            # Calculate cosine similarity between the embeddings
            embeddings1 = model.embedding(encoded_sent1['input_ids'])
            embeddings2 = model.embedding(encoded_sent2['input_ids'])
            cos_sim = cosine_similarity(embeddings1.mean(dim=1), embeddings2.mean(dim=1), dim=1)
            cosine_similarities.extend(cos_sim.cpu().numpy())
    
    return np.array(predictions), np.array(true_scores), np.array(cosine_similarities)

# Main function for evaluating metrics
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
    
    BATCH_SIZE = 64  # Reduced for computational feasibility
    MAX_LENGTH = 128

    # Load test data
    test_loader = prepare_dataloader(tokenizer, MAX_LENGTH, BATCH_SIZE, device)

    # Load model
    vocab_size = tokenizer.vocab_size
    model = LSTMModel(vocab_size, EMBED_DIM, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS, BIDIRECTIONAL, DROPOUT)
    model.load_state_dict(torch.load('./outputs/lstm_model.pt'))
    model = model.to(device)

    # Evaluate the model
    predictions, true_scores, cosine_similarities = evaluate_model(model, test_loader, device)

    # Calculate Pearson Correlation
    pearson_corr, _ = scipy.stats.pearsonr(predictions, true_scores)
    print(f'Pearson Correlation: {pearson_corr:.3f}')

    # Calculate Mean Squared Error using the revised method
    mse_value = mse(true_scores, predictions)
    print(f'Mean Squared Error: {mse_value:.3f}')

    # Calculate R^2 Score
    r2 = r2_score(true_scores, predictions)
    print(f'R^2 Score: {r2:.3f}')

    # Calculate Average Cosine Similarity
    avg_cosine_similarity = np.mean(cosine_similarities)
    print(f'Average Cosine Similarity: {avg_cosine_similarity:.3f}')

    # Save metrics to outputs directory
    output_dir = './outputs'
    os.makedirs(output_dir, exist_ok=True)
    results = {
        'pearson_correlation': pearson_corr,
        'mean_squared_error': mse_value,
        'r2_score': r2,
        'average_cosine_similarity': avg_cosine_similarity
    }
    with open(f'{output_dir}/evaluation_metrics.json', 'w') as f:
        json.dump(results, f)

if __name__ == '__main__':
    main()
