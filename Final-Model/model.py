import torch
from torch import nn

class Model(nn.Module):
    def __init__(self, max_len, bert, hidden_dim, no_layers, single_token_output=True):
        super(Model, self).__init__()
        self.bert = bert
        self.hidden_dim = hidden_dim
        self.embedding_dim = bert.config.to_dict()['hidden_size']
        self.num_layers = no_layers
        self.n_vocab = bert.config.to_dict()['vocab_size']
        self.max_len = max_len
        self.lstm = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=0.2,
        )
        # self.fc = nn.Linear(self.hidden_dim, self.n_vocab)
        # self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(self.hidden_dim, 256)
        self.fc2 = nn.Linear(256, self.n_vocab)
        # self.fc2 = nn.Linear(256, 512)
        # self.fc3 = nn.Linear(512, self.n_vocab)
        self.single_token_output = single_token_output
        self.act = nn.ReLU()

    def forward(self, x, hidden, x_attention):
        embed = self.bert(input_ids=x, attention_mask=x_attention)[0]
        output, hidden = self.lstm(embed, hidden)
        out = self.fc1(output)
        out = self.fc2(out)
        # out = self.fc(output)
        # out = self.act(self.fc1(output))
        # out = self.act(self.fc2(out))
        # out = self.fc3(out)
        if self.single_token_output is True:
            out = out[:, -1, :]  # keeps only last logits, i.e. logits associated with the last word we want to predict
        # out = self.softmax(out)
        return out, hidden

    def init_hidden(self, batch_size):
        h0 = torch.zeros((self.num_layers, batch_size, self.hidden_dim)).to(device)
        c0 = torch.zeros((self.num_layers, batch_size, self.hidden_dim)).to(device)
        return h0, c0
