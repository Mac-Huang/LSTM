import torch
from torch.utils.data import Dataset

class STSBDataset(Dataset):
    def __init__(self, sentences1, sentences2, scores, tokenizer, max_length, device):
        self.sentences1 = sentences1
        self.sentences2 = sentences2
        self.scores = scores
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = device

    def __len__(self):
        return len(self.sentences1)

    def __getitem__(self, idx):
        sent1 = self.sentences1[idx]
        sent2 = self.sentences2[idx]
        score = self.scores[idx]

        encoded_sent1 = self.tokenizer(sent1, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt')
        encoded_sent2 = self.tokenizer(sent2, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt')

        # Calculate the sequence lengths from the attention mask
        length1 = torch.sum(encoded_sent1['attention_mask'])
        length2 = torch.sum(encoded_sent2['attention_mask'])
        
        input1 = {key: val.squeeze(0).to(self.device) for key, val in encoded_sent1.items()}
        input2 = {key: val.squeeze(0).to(self.device) for key, val in encoded_sent2.items()}

        # Return the inputs, score, and lengths of both sentences
        return (input1, input2), torch.tensor(score, dtype=torch.float, device=self.device), (length1, length2)
