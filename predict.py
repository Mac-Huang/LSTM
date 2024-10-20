import os
import torch
import json
from src.model import LSTMModel
from transformers import BertTokenizer

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the vocab from the JSON file
vocab_file_path = './outputs/vocab.json'

print("Loading vocabulary...")
if os.path.exists(vocab_file_path):
    with open(vocab_file_path, 'r') as f:
        vocab_obj = json.load(f)
else:
    raise FileNotFoundError(f"Vocabulary file not found at {vocab_file_path}")

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained('./bert-base-uncased')

# Define text pipeline for prediction
def text_pipeline(text):
    tokens = tokenizer.tokenize(text)
    return [vocab_obj.get(token, vocab_obj.get("[UNK]")) for token in tokens]

# Set hyperparameters
EMBED_DIM = 1024
HIDDEN_DIM = 1024
OUTPUT_DIM = 1
N_LAYERS = 5
BIDIRECTIONAL = True
DROPOUT = 0.5

BATCH_SIZE = 1024
EPOCHS = 10
LR = 5e-4
MAX_LENGTH = 256

# Define model parameters (should match those used during training)
vocab_size = len(vocab_obj)
embedding_dim = EMBED_DIM
hidden_dim = HIDDEN_DIM
output_dim = OUTPUT_DIM
n_layers = N_LAYERS
bidirectional = BIDIRECTIONAL
dropout_rate = DROPOUT

# Load the model
print("Loading model...")
model = LSTMModel(vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout_rate)
model.load_state_dict(torch.load('./outputs/lstm_model.pt', map_location=device))
model = model.to(device)
model.eval()

# Function to predict sentiment
def predict_sentiment(review):
    with torch.no_grad():
        processed_text = torch.tensor(text_pipeline(review)).unsqueeze(0).to(device)
        length = torch.tensor([len(processed_text[0])], dtype=torch.int64).to("cpu")  # length should be on CPU for pack_padded_sequence
        prediction = model(processed_text, length)
        prediction = torch.sigmoid(prediction).item()  # Use sigmoid since BCEWithLogitsLoss was used
        return 'positive' if prediction >= 0.5 else 'negative'
    
# Function to calculate the model size
def calculate_model_size(model):
    total_params = 0
    total_size = 0  # in bytes
    for param in model.parameters():
        param_size = param.numel() * param.element_size()  # numel() returns the total number of elements in the tensor, element_size() returns the size in bytes
        total_params += param.numel()
        total_size += param_size
    
    total_params = total_params / 1e6
    total_size = total_size / (1024 ** 2)  # Convert bytes to megabytes
    print(f"Total Parameters: {total_params:.2f} M")
    print(f"Model Size: {total_size:.2f} MB")

# Example usage
if __name__ == "__main__":
    # Calculate and print model size
    calculate_model_size(model)
    
    test_reviews = [
        "The movie is wonderfully enjoyable!",  # Positive, should be predicted correctly
        "The plot was terrible but the actors did a great job.",  # Mixed, tricky to predict
        "I didn't hate it, but I wouldn't watch it again.",  # Neutral sentiment, could go either way
        "The visuals were stunning, but the story lacked depth and the characters were flat.",  # Mixed, leans negative
        "It's not the worst film I've ever seen, but it was quite boring.",  # Negative, subtly expressed
        "Absolutely phenomenal! Best movie of the year, hands down.",  # Strong positive, should be correct
        "This movie was a disaster from start to finish, I wish I could get my time back.",  # Strong negative
        "Some parts were okay, but overall, I found it to be underwhelming and forgettable.",  # Mixed to negative
        "The acting was decent, but the script was filled with clichés, making it hard to enjoy.",  # Negative despite some positive elements
        "It was just fine. Nothing stood out, but nothing was terrible either.",  # Ambiguous, challenging to classify
        "The film's intentions were good, but the execution left much to be desired.",  # Negative but diplomatically stated
        "One of the worst movies I have ever seen, yet I couldn't help but admire the effort put in by the cast.",  # Mixed but leans negative
        "The movie was confusing and poorly edited, but it had a couple of good action scenes.",  # Mixed but leans negative
        "I laughed a few times, but the rest of the film was just awkward and badly paced.",  # Negative with small positive moments
        "The movie started strong but fell apart completely by the end.",  # Negative, despite initial positivity
        "The director's vision was ambitious, and some scenes worked, but ultimately, it missed the mark.",  # Mixed sentiment, difficult to predict
        "A breathtakingly beautiful movie with a dull and uninspired storyline.",  # Visually positive, overall negative
        "It was an average film, but the soundtrack was truly spectacular.",  # Positive and negative components
        "I can't say I enjoyed it, but there were a few redeeming qualities here and there.",  # Mixed, leaning negative
        "The film had a lot of promise, but the execution was simply lackluster.",  # Negative sentiment
        "Fantastic visuals and a brilliant score, but the characters were poorly written and unrelatable.",  # Positive and negative balance
        "An absolute trainwreck of a movie, though the ending did provide a small bit of redemption.",  # Strong negative with a small positive
        "It had potential, but unfortunately, it was just wasted on a weak script and subpar direction.",  # Negative, with some hint of potential
    ]

    for review in test_reviews:
        sentiment = predict_sentiment(review)
        print(f'Review: "{review}"')
        print(f'Review Sentiment: {sentiment}\n')
