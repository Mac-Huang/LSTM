import os
import nltk
import pandas as pd
import torch
from torch.utils.data import Dataset
from nltk.corpus import stopwords
import re

# # Load stopwords once (FAIL)
# english_stops = set(stopwords.words('english'))

# I have to manually download and set it
current_dir = os.path.dirname(__file__)
stopwords_path = os.path.join(current_dir, '..', 'data', 'stopwords', 'english')
if not os.path.isfile(stopwords_path):
    raise FileNotFoundError(f"Error: Stopwords file not found at path: {stopwords_path}")
with open(stopwords_path, 'r', encoding='utf-8') as f:
    english_stops = set(f.read().splitlines())

# Compile regex patterns for better performance
html_pattern = re.compile(r'<br\s*/?>|<.*?>')  # Block <div> & <br>
non_alphabet_pattern = re.compile(r'[^A-Za-z!?]')  # Keep alphabet and special characters
add_space = re.compile(r'([!?])')  # Add a space before ! and ? characters

class IMDBDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length, device='cpu', batch_tokenize=False):
        """
        Initialize the dataset.

        Args:
        - csv_file (str): Path to the CSV file containing the dataset.
        - tokenizer (callable): A tokenizer that converts text into sequences.
        - max_length (int): Maximum length for padding/truncating sequences.
        - device (str): Device to load data onto ('cpu' or 'cuda').
        - batch_tokenize (bool): Option to tokenize data in batches for better performance.
        """
        self.df = pd.read_csv(csv_file)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = device  # Store the device for moving tensors
        self.x_data = self.df['review']
        self.y_data = self.df['sentiment'].apply(lambda x: 1 if x == 'positive' else 0)
        self.clean_data()

        # Batch tokenization flag
        self.batch_tokenize = batch_tokenize
        if self.batch_tokenize:
            self.tokenize_all_reviews()

    def clean_data(self):
        """
        Clean the review text data by applying pre-processing steps.
        """
        self.x_data = self.x_data.apply(self.preprocess)

    def preprocess(self, text):
        """
        Apply preprocessing steps to a single review.
        """
        text = html_pattern.sub('', text)  # Remove HTML tags
        text = non_alphabet_pattern.sub(' ', text)  # Remove non-alphabet characters
        text = add_space.sub(r' \1', text)  # Add a space before ! and ? characters
        words = text.split()
        words = [w.lower() for w in words if w not in english_stops]  # Lowercase and remove stop words
        return ' '.join(words)

    def tokenize_all_reviews(self):
        """
        Tokenize all reviews at once for better performance if batch_tokenize is True.
        This can reduce overhead when working with large datasets.
        """
        self.x_data = self.tokenizer(
            list(self.x_data),
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )['input_ids'].to(self.device)

    def __len__(self):
        """
        Return the number of samples in the dataset.
        """
        return len(self.df)

    def __getitem__(self, idx):
        """
        Retrieve the tokenized input and corresponding sentiment label.

        Args:
        - idx (int): Index of the sample to retrieve.

        Returns:
        - tokens (torch.Tensor): Tokenized and padded/truncated review.
        - sentiment (torch.Tensor): Sentiment label (0 for negative, 1 for positive).
        """
        if self.batch_tokenize:
            # If batch tokenization is used, simply retrieve the pre-tokenized review
            tokens = self.x_data[idx]
        else:
            review = self.x_data.iloc[idx]
            tokens = self.tokenizer(
                review,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )['input_ids'].squeeze().to(self.device)
        
        sentiment = torch.tensor(self.y_data.iloc[idx], dtype=torch.long).to(self.device)
        length = min(len(review.split()), self.max_length)
        
        return tokens, sentiment, length
