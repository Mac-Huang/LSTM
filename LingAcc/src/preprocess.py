import os
import pandas as pd
import torch
from torch.utils.data import Dataset
import re

# Manually set the path for stopwords
current_dir = os.path.dirname(__file__)
stopwords_path = os.path.join(current_dir, '..', '..', 'data', 'stopwords', 'english')
if not os.path.isfile(stopwords_path):
    raise FileNotFoundError(f"Error: Stopwords file not found at path: {stopwords_path}")
with open(stopwords_path, 'r', encoding='utf-8') as f:
    english_stops = set(f.read().splitlines())

# Compile regex patterns for performance optimization
html_pattern = re.compile(r'<br\s*/?>|<.*?>')  # Remove HTML tags
non_alphabet_pattern = re.compile(r'[^A-Za-z!?]')  # Keep only alphabet and special characters
add_space = re.compile(r'([!?])')  # Add space before ! and ? characters

class CoLADataset(Dataset):
    def __init__(self, tsv_file, tokenizer, max_length, device='cpu', batch_tokenize=False):
        """
        Initialize the dataset class

        Args:
        - tsv_file (str): Path to the CoLA dataset file (.tsv format)
        - tokenizer (callable): Tokenizer to convert text into sequences
        - max_length (int): Maximum length for padding/truncating sequences
        - device (str): Device to load the data onto ('cpu' or 'cuda')
        - batch_tokenize (bool): Option for batch tokenization for better performance
        """
        self.df = pd.read_csv(tsv_file, sep='\t', header=None, names=['source', 'label', 'original_label', 'sentence'])
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = device
        self.x_data = self.df['sentence']
        self.y_data = self.df['label']
        self.clean_data()

        # Batch tokenization flag
        self.batch_tokenize = batch_tokenize
        if self.batch_tokenize:
            self.tokenize_all_sentences()

    def clean_data(self):
        """
        Clean the sentence data by applying pre-processing steps
        """
        self.x_data = self.x_data.apply(self.preprocess)

    def preprocess(self, text):
        """
        Preprocess a single sentence

        Args:
        - text (str): Input sentence

        Returns:
        - cleaned_text (str): Cleaned sentence
        """
        text = html_pattern.sub('', text)  # Remove HTML tags
        text = non_alphabet_pattern.sub(' ', text)  # Remove non-alphabet characters
        text = add_space.sub(r' \1', text)  # Add space before ! and ? characters
        words = text.split()
        words = [w.lower() for w in words if w not in english_stops]  # Lowercase and remove stopwords
        return ' '.join(words)

    def tokenize_all_sentences(self):
        """
        Pre-tokenize all sentences at once for better performance if batch_tokenize is True
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
        Return the number of samples in the dataset
        """
        return len(self.df)

    def __getitem__(self, idx):
        """
        Retrieve the tokenized input and corresponding label

        Args:
        - idx (int): Index of the sample to retrieve

        Returns:
        - tokens (torch.Tensor): Tokenized and padded/truncated sentence
        - label (torch.Tensor): Acceptability label (0 for unacceptable, 1 for acceptable)
        - length (int): Actual length of the sentence
        """
        if self.batch_tokenize:
            tokens = self.x_data[idx]
        else:
            sentence = self.x_data.iloc[idx]
            tokens = self.tokenizer(
                sentence,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )['input_ids'].squeeze().to(self.device)

        label = torch.tensor(self.y_data.iloc[idx], dtype=torch.long).to(self.device)
        length = min(len(sentence.split()), self.max_length)
        
        return tokens, label, length
