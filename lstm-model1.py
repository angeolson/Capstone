# Initial Model based on tutorial from https://closeheat.com/blog/pytorch-lstm-text-generation-tutorial

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

# -----------HELPER FUNCTIONS------------
def cleanData(df):
    '''
    function cleans the original pandas dataframe imported from .csv and returns a clean frame
    :param df:
    :return: cleaned pandas dataframe
    '''
    for col in ['verses', 'verse_types', 'verses_transformed', 'EDA_verses']:
        df[col] = df[col].apply(lambda x:eval(x, {'__builtins__': None}, {}))
    return df

def tokenizer(x):
    return x.split(" ")


#---------SET VARS--------------------
EPOCHS = 2
BATCH_SIZE = 512
SEQUENCE_LEN = 4
MAX_LEN = 210
#--------CLASS DEFINITIONS-------------

class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataframe,
        sequence_length,
        tokenizer,
        batch_size,
        max_len
    ):
        self.dataframe = dataframe
        self.words = self.load_words(self.dataframe)
        self.uniq_words = self.get_uniq_words()
        self.index_to_word = {index: word for index, word in enumerate(self.uniq_words)}
        self.word_to_index = {word: index for index, word in enumerate(self.uniq_words)}
        self.words_indexes = [self.word_to_index[w] for w in self.words]
        self.sequence_length = sequence_length
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_len = max_len

    def load_words(self, dataframe):
        text = dataframe['lyrics'].str.cat(sep=' ')
        return text.split(' ')
    def get_uniq_words(self):
        word_counts = Counter(self.words)
        unique_words = sorted(word_counts, key=word_counts.get, reverse=True)
        unique_words_padding = ['PAD'] + unique_words
        return unique_words_padding

    def __len__(self):
        return len(self.dataframe)

    def build_sequences(self, text, word_to_index, sequence_length, max_len):
        '''
        adapted from https://gist.github.com/FernandoLpz/acaeb5fe714d084c0fe08481fdaa08b7#file-build_sequences-py
        :param text:
        :param word_to_index:
        :param sequence_length:
        :return:
        '''
        x = list()
        y = list()

        for i in range(max_len):
            try:
                # Get window of chars from text
                # Then, transform it into its idx representation
                sequence = text[i:i + sequence_length]
                sequence = [word_to_index[char] for char in sequence]

                # Get word target
                # Then, transfrom it into its idx representation
                target = text[i + sequence_length]
                target = word_to_index[target]

                # Save sequences and targets
                x.append(sequence)
                y.append(target)

            except:
                sequence = [0]*sequence_length
                x.append(sequence)
                y.append(0)

        x = np.array(x)
        y = np.array(y)

        return x, y

    # def __data_generation(self, sequences_batch, next_words_batch):
    #     X = np.zeros((self.batch_size, self.sequence_length, len(self.uniq_words)), dtype=np.bool)
    #     y = np.zeros((self.batch_size, len(self.uniq_words)), dtype=np.bool)
    #
    #     for i, seq in enumerate(sequences_batch):
    #         for j, word in enumerate(seq):
    #             X[i, j, word] = 1
    #             y[i, next_words_batch[i]] = 1
    #     return X, y

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        input = self.dataframe.iloc[idx]['lyrics']
        tokenized_input = self.tokenizer(input)
        X, y = self.build_sequences(text=tokenized_input, word_to_index=self.word_to_index, sequence_length=self.sequence_length, max_len=self.max_len)

        return torch.tensor(X), torch.tensor(y)

class Model(nn.Module):
    def __init__(self, dataset):
        super(Model, self).__init__()
        self.hidden_dim = 128
        self.embedding_dim = 128
        self.num_layers = 3
        n_vocab = len(dataset.uniq_words)
        self.embedding = nn.Embedding(
            num_embeddings=n_vocab,
            embedding_dim=self.embedding_dim,
        )
        self.lstm = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=0.2,
            batch_fist=True
        )
        self.fc = nn.Linear(self.lstm_size, n_vocab)

    def forward(self, x, hidden):
        embed = self.embedding(x)
        output, hidden = self.lstm(embed, hidden)
        out = self.fc(output)
        return out, hidden
    def forward(self, x, hidden, attention_mask):
        batch_size = x.size(0)
        embedded = self.bert(input_ids=x,attention_mask=attention_mask)[0]
        #input = embedded.permute(1, 0, 2)
        lstm_out, hidden = self.lstm(embedded, hidden)
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)
        out = self.dropout(lstm_out)
        out = self.fc(out)
        out = out.view(batch_size, -1, self.output_dim)
        out = out[:, -1]
        return out, hidden
    def init_state(self, sequence_length):
        return (torch.zeros(self.num_layers, sequence_length, self.lstm_size),
                torch.zeros(self.num_layers, sequence_length, self.lstm_size))

    def init_hidden(self, batch_size, ):
        h0 = torch.zeros((self.no_layers, batch_size, self.hidden_dim)).to(device)
        c0 = torch.zeros((self.no_layers, batch_size, self.hidden_dim)).to(device)
        return h0, c0

#--------------MODEL FUNCTIONS----------------
def train(dataset, model, batch_size, sequence_length, max_epochs):
    model.train()
    dataloader = DataLoader(dataset, batch_size=batch_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(max_epochs):
        state_h, state_c = model.init_state(sequence_length)
        #for batch, (x, y) in enumerate(dataloader):
        for batch in dataloader:
            optimizer.zero_grad()
            x = batch[0]
            y = batch[1]
            y_pred, (state_h, state_c) = model(x, (state_h, state_c))
            loss = criterion(y_pred.transpose(1, 2), y)
            state_h = state_h.detach()
            state_c = state_c.detach()
            loss.backward()
            optimizer.step()
            print({ 'epoch': epoch, 'batch': batch, 'loss': loss.item() })

def predict(dataset, model, text, next_words=100):
    model.eval()
    words = text.split(' ')
    state_h, state_c = model.init_state(len(words))
    for i in range(0, next_words):
        x = torch.tensor([[dataset.word_to_index[w] for w in words[i:]]])
        y_pred, (state_h, state_c) = model(x, (state_h, state_c))
        last_word_logits = y_pred[0][-1]
        p = torch.nn.functional.softmax(last_word_logits, dim=0).detach().numpy()
        word_index = np.random.choice(len(last_word_logits), p=p)
        words.append(dataset.index_to_word[word_index])
    return words

#-------------MODEL PREP----------------
df = pd.read_csv('df_LSTM.csv', index_col=0)
df.reset_index(drop=True, inplace=True)
dataset = Dataset(dataframe=df, sequence_length=SEQUENCE_LEN, tokenizer=tokenizer, batch_size=BATCH_SIZE, max_len=MAX_LEN)
model = Model(dataset)


# testing
# dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)
# for batch in dataloader:
#     break

#------------MODEL TRAIN----------------
train(dataset, model, batch_size=BATCH_SIZE, sequence_length=SEQUENCE_LEN, max_epochs=EPOCHS)

#------------MODEL RUN-----------------
print(predict(dataset, model, text='love'))