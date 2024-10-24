import torch

class LSTMModel(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout_rate, pad_index=0):
        super(LSTMModel, self).__init__()
        
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_index)
        self.lstm = torch.nn.LSTM(embedding_dim, hidden_dim, n_layers, bidirectional=bidirectional, dropout=dropout_rate, batch_first=True)
        self.fc = torch.nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.layer_norm = torch.nn.LayerNorm(hidden_dim * 2 if bidirectional else hidden_dim)

    def forward(self, ids, lengths):

        embedded = self.dropout(self.embedding(ids))

        # Pack the sequence to handle varying sequence lengths in the batch
        packed_embedded = torch.nn.utils.rnn.pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)

        # Passing the packed sequences through the LSTM layer
        packed_output, (hidden, cell) = self.lstm(packed_embedded)

        # We only care about the hidden states from the last layer
        if self.lstm.bidirectional:
            hidden = torch.cat((hidden[-2], hidden[-1]), dim=-1)  # Forward and backward hidden states
        else:
            hidden = hidden[-1]

        hidden = self.dropout(hidden)
        hidden = self.layer_norm(hidden)
        output = self.fc(hidden)

        return output
