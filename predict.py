import torch
from transformers import BertTokenizer
from src.model import LSTMModel
import sys

# Set device
def set_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize tokenizer
def load_tokenizer():
    return BertTokenizer.from_pretrained('../bert-base-uncased')

# Load the model
def load_model(device, tokenizer, model_path):
    EMBED_DIM = 1536
    HIDDEN_DIM = 768
    OUTPUT_DIM = 1  # Corrected to scalar output for regression task
    N_LAYERS = 2
    BIDIRECTIONAL = True
    DROPOUT = 0.2

    vocab_size = tokenizer.vocab_size
    model = LSTMModel(vocab_size, EMBED_DIM, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS, BIDIRECTIONAL, DROPOUT)
    state_dict = torch.load(model_path)
    model.load_state_dict({k: v for k, v in state_dict.items() if k in model.state_dict()})
    model = model.to(device)
    model.eval()
    return model

# Predict similarity between two sentences
def predict_similarity(model, tokenizer, sentence1, sentence2, device, max_length=128):
    model.eval()
    encoded_sent1 = tokenizer(sentence1, padding='max_length', truncation=True, max_length=max_length, return_tensors='pt')
    encoded_sent2 = tokenizer(sentence2, padding='max_length', truncation=True, max_length=max_length, return_tensors='pt')

    encoded_sent1 = {k: v.to(device) for k, v in encoded_sent1.items()}
    encoded_sent2 = {k: v.to(device) for k, v in encoded_sent2.items()}
    length = torch.tensor([max_length]).to(device)

    with torch.no_grad():
        prediction = model(encoded_sent1['input_ids'], encoded_sent2['input_ids'], length)
    
    return prediction.item()

# Main function for predicting similarity
def main():
    # Set up device and tokenizer
    device = set_device()
    tokenizer = load_tokenizer()

    # Load the model
    model_path = './outputs/lstm_model.pt'
    model = load_model(device, tokenizer, model_path)

    # Get input sentences from user
    print("Enter the first sentence:")
    sentence1 = input().strip()
    print("Enter the second sentence:")
    sentence2 = input().strip()

    # Predict similarity
    similarity_score = predict_similarity(model, tokenizer, sentence1, sentence2, device)
    print(f'Similarity Score: {similarity_score:.3f}')

if __name__ == '__main__':
    main()
