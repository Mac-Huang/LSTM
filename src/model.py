import torch
import torch.nn.functional as F

class LSTMModel(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout_rate, pad_index=0):
        super(LSTMModel, self).__init__()
        
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_index)
        self.lstm = torch.nn.LSTM(embedding_dim * 2, hidden_dim, n_layers, bidirectional=bidirectional, dropout=dropout_rate, batch_first=True)
        self.fc = torch.nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.layer_norm = torch.nn.LayerNorm(hidden_dim * 2 if bidirectional else hidden_dim)
        self.highway = torch.nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, hidden_dim * 2 if bidirectional else hidden_dim)
        
    def forward(self, input1, input2, lengths):
        embed1 = self.embedding(input1)
        embed2 = self.embedding(input2)
        # print(embed1.shape)
        
        combined_embed = self.dropout(torch.cat((embed1, embed2), dim=2))
        # print(combined_embed.shape)
        
        packed_embed = torch.nn.utils.rnn.pack_padded_sequence(combined_embed, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, (hidden, cell) = self.lstm(packed_embed)
        output_unpacked, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        # print(hidden.shape)
        
        self.lstm.flatten_parameters()

        # Concatenate the hidden states if LSTM is bidirectional, otherwise take the last hidden state
        if self.lstm.bidirectional:
            hidden = torch.cat((hidden[-2], hidden[-1]), dim=-1)
        else:
            hidden = hidden[-1]
            
        # print(hidden.shape)
        
        # Apply dropout and layer normalization to the hidden state
        hidden = self.dropout(hidden)
        # print(hidden.shape)
        hidden = self.layer_norm(hidden)
        # print(hidden.shape)
        output = self.fc(hidden)

        return output
