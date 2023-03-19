# imports
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from collections import Counter
import numpy as np
import random

SEED = 48
random.seed(48)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MAX_LEN = 350

#--------MODEL DEFINITION-------------
class Model(nn.Module):
    def __init__(self, uniq_words, max_len):
        super(Model, self).__init__()
        self.hidden_dim = 128
        self.embedding_dim = 128
        self.num_layers = 3
        # n_vocab = len(dataset.uniq_words)
        # self.max_len = dataset.max_len
        n_vocab = len(uniq_words)
        self.max_len = max_len
        self.embedding = nn.Embedding(
            num_embeddings=n_vocab,
            embedding_dim=self.embedding_dim,
        )
        self.lstm = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=0.2,
            batch_first=True
        )
        self.fc = nn.Linear(self.hidden_dim, n_vocab)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, hidden):
        # batch_size = x.size(0)
        embed = self.embedding(x)
        output, hidden = self.lstm(embed, hidden)
        out = self.fc(output)
        out = out[:, -1, :] # keeps only last subtensor tensor; likely want to use attention mechanism to create linear combo of all
        out = self.softmax(out)
        return out, hidden

    def init_hidden(self, batch_size):
        h0 = torch.zeros((self.num_layers, batch_size, self.hidden_dim)).to(device)
        c0 = torch.zeros((self.num_layers, batch_size, self.hidden_dim)).to(device)
        return h0, c0

# -----------HELPER FUNCTIONS------------
def predict(word_to_index, index_to_word, model, text, next_words=250):
    model.eval()
    words = text.split(' ')
    state_h, state_c = model.init_hidden(1)
    for i in range(0, next_words):
        x = torch.tensor([[word_to_index[w] for w in words[i:]]]).to(device)
        y_pred, (state_h, state_c) = model(x, (state_h, state_c))
        last_word_logits = y_pred[0]
        p = last_word_logits.detach().cpu().numpy()
        word_index = np.random.choice(len(last_word_logits), p=p)
        words.append(index_to_word[word_index])
    return words

def load_words(dataframe):
    text = dataframe['lyrics'].str.cat(sep=' ')
    return text.split(' ')

def get_uniq_words(words):
    word_counts = Counter(words)
    unique_words = sorted(word_counts, key=word_counts.get, reverse=True)
    unique_words_padding = ['PAD', '<NEWSONG>'] + unique_words
    return unique_words_padding


#---------LOAD DATA--------------------
df = pd.read_csv('df_LSTM.csv', index_col=0)
df_copy = df.copy()
df_copy.reset_index(drop=True, inplace=True)
df_copy = df.iloc[0:700]

# create word dictionary for all datasets
all_words = load_words(df_copy)
uniq_words = get_uniq_words(all_words)

index_to_word = {index: word for index, word in enumerate(uniq_words)}
word_to_index = {word: index for index, word in enumerate(uniq_words)}
words_indexes = [word_to_index[w] for w in all_words]

#---------LOAD MODEL--------------------
model = Model(uniq_words=uniq_words, max_len=MAX_LEN).to(device)
model.load_state_dict(torch.load('model_1.pt', map_location=device))

#------------MODEL RUN-----------------
print(predict(word_to_index=word_to_index, index_to_word=index_to_word, model=model, text='hey', next_words=250))