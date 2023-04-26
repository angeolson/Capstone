# imports
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from collections import Counter
import numpy as np
import random
import torch.nn.functional as F

SEED = 48
random.seed(48)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MAX_LEN = 300
glove = False

embedding_dim = 200  # set = 50 for the 50d file, eg.
filepath = f'Glove/glove.6B.{embedding_dim}d.txt'  # set filepath
single_token_output=False
save_model = f'model-1-all.py'

#----HELPER FUNCTIONS-------
def embedding_for_vocab(filepath, word_index,
                        embedding_dim):
    vocab_size = len(word_index)

    # Adding again 1 because of reserved 0 index
    embedding_matrix_vocab = np.zeros((vocab_size,
                                       embedding_dim))

    with open(filepath, encoding="utf8") as f:
        for line in f:
            word, *vector = line.split()
            if word in word_index:
                idx = word_index[word]
                embedding_matrix_vocab[idx] = np.array(
                    vector, dtype=np.float32)[:embedding_dim]

    return embedding_matrix_vocab

#--------MODEL DEFINITION-------------
class Model(nn.Module):
    def __init__(self, uniq_words, max_len, embedding_dim, embedding_matrix, single_token_output=True):
        super(Model, self).__init__()
        self.hidden_dim = 128
        self.embedding_dim = embedding_dim
        self.num_layers = 4
        n_vocab = len(uniq_words)
        self.max_len = max_len
        if glove is True:
            self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(embedding_matrix), padding_idx=0)
        else:
            self.embedding = nn.Embedding(
            num_embeddings=n_vocab,
            embedding_dim=self.embedding_dim,
            padding_idx=0
        )
        self.lstm = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=0.2,
            batch_first=True
        )
        self.fc = nn.Linear(self.hidden_dim, n_vocab)
        self.single_token_output = single_token_output

    def forward(self, x, hidden):
        embed = self.embedding(x)
        output, hidden = self.lstm(embed, hidden)
        out = self.fc(output)
        if self.single_token_output is True:
            out = out[:, -1, :] # keeps only last logits, i.e. logits associated with the last word we want to predict
        #out = self.softmax(out)
        return out, hidden

    def init_hidden(self, batch_size):
        h0 = torch.zeros((self.num_layers, batch_size, self.hidden_dim)).to(device)
        c0 = torch.zeros((self.num_layers, batch_size, self.hidden_dim)).to(device)
        return h0, c0

# -----------HELPER FUNCTIONS------------
def predict(word_to_index, index_to_word, model, text, single_token_output, next_words=250):
    model.eval()
    words = [ ]
    for item in text.split(' '):
        words.append(item)
    state_h, state_c = model.init_hidden(1)
    for i in range(0, next_words):
        x = torch.tensor([[word_to_index[w] for w in words[i:]]][-4:]).to(device)
        #x = torch.tensor([[word_to_index[w] for w in words[i:]]]).to(device)
        y_pred, (state_h, state_c) = model(x, (state_h, state_c))
        if single_token_output is True:
            logits = y_pred[0]
        else:
            logits = y_pred[0][-1]
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        sorted_logits_prob = F.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_logits_prob, dim=-1)
        sorted_indices_to_remove = cumulative_probs > 0.8
        keep = sorted_indices[sorted_indices_to_remove]
        sorted_logits_prob_keep = sorted_logits_prob[:len(keep)]
        if len(sorted_logits_prob_keep) == 0:
            next_token = 0 # padding token
        else:
            next_token_sorted = torch.multinomial(sorted_logits_prob, num_samples=1)
            next_token = keep[next_token_sorted].detach().cpu().numpy()[0]
        # softmax = nn.Softmax(dim=0)
        # word_softmax = softmax(logits)
        # p = word_softmax.detach().cpu().numpy()
        # word_index = np.random.choice(len(last_word_logits), p=p)
        # words.append(index_to_word[word_index])
        words.append(index_to_word[next_token])
    return " ".join(words)

def load_words(dataframe):
    text = dataframe['lyrics'].str.cat(sep=' ')
    return text.split(' ')

def get_uniq_words(words):
    word_counts = Counter(words)
    unique_words = sorted(word_counts, key=word_counts.get, reverse=True)
    unique_words_padding = ['<PAD>'] + unique_words
    return unique_words_padding


#---------LOAD DATA--------------------
df = pd.read_csv('df_LSTM.csv', index_col=0)
df_copy = df.copy()
df_copy.reset_index(drop=True, inplace=True)

# create word dictionary for all datasets
all_words = load_words(df_copy)
uniq_words = get_uniq_words(all_words)

index_to_word = {index: word for index, word in enumerate(uniq_words)}
word_to_index = {word: index for index, word in enumerate(uniq_words)}
words_indexes = [word_to_index[w] for w in all_words]
embedding_matrix = embedding_for_vocab(filepath=filepath, word_index=word_to_index, embedding_dim=embedding_dim)

#---------LOAD MODEL--------------------
model = Model(uniq_words=uniq_words, max_len=MAX_LEN, single_token_output=single_token_output, embedding_dim=embedding_dim, embedding_matrix=embedding_matrix).to(device)
model.load_state_dict(torch.load(save_model, map_location=device))

#------------MODEL RUN-----------------
print(predict(word_to_index=word_to_index, index_to_word=index_to_word, model=model, text="you", next_words=250, single_token_output=single_token_output))