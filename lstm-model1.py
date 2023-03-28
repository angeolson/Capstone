# Initial Model based on tutorial from https://closeheat.com/blog/pytorch-lstm-text-generation-tutorial

# imports
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from collections import Counter
import numpy as np
import random
from sklearn.model_selection import train_test_split

SEED = 48
random.seed(48)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
glove = True

#---------SET VARS--------------------
EPOCHS = 5
MAX_LEN = 350
SEQUENCE_LEN = 4
BATCH_SIZE = MAX_LEN - SEQUENCE_LEN

embedding_dim = 100  # set = 50 for the 50d file, eg.
filepath = f'Glove/glove.6B.{embedding_dim}d.txt'  # set filepath



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

def load_words(dataframe):
    text = dataframe['lyrics'].str.cat(sep=' ')
    return text.split(' ')
def get_uniq_words(words):
    word_counts = Counter(words)
    unique_words = sorted(word_counts, key=word_counts.get, reverse=True)
    unique_words_padding = ['PAD', '<NEWSONG>'] + unique_words
    return unique_words_padding

# code for embedding function adapted from https://www.geeksforgeeks.org/pre-trained-word-embedding-using-glove-in-nlp-models/
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

#--------CLASS DEFINITIONS-------------

class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataframe,
        sequence_length,
        tokenizer,
        batch_size,
        max_len,
        words,
        uniq_words,

    ):
        self.dataframe = dataframe
        # self.words = self.load_words(self.dataframe)
        # self.uniq_words = self.get_uniq_words()
        self.uniq_words = uniq_words
        self.words = words
        self.index_to_word = {index: word for index, word in enumerate(self.uniq_words)}
        self.word_to_index = {word: index for index, word in enumerate(self.uniq_words)}
        self.words_indexes = [self.word_to_index[w] for w in self.words]
        # self.embedding_matrix = embedding_for_vocab(filepath=filepath, word_index=self.word_to_index, embedding_dim=embedding_dim)
        self.sequence_length = sequence_length
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_len = max_len
        self.inputs, self.targets = self.get_data(self.dataframe)

    # def load_words(self, dataframe):
    #     text = dataframe['lyrics'].str.cat(sep=' ')
    #     return text.split(' ')
    # def get_uniq_words(self):
    #     word_counts = Counter(self.words)
    #     unique_words = sorted(word_counts, key=word_counts.get, reverse=True)
    #     unique_words_padding = ['PAD'] + unique_words
    #     return unique_words_padding

    def __len__(self):
        return len(self.inputs)

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
        text_length = len(text)
        if text_length < max_len:
            for i in range(max_len-text_length):
                text.append('PAD')
        for i in range(max_len - (sequence_length + 1)):
            # try:
            # Get window of chars from text
            # Then, transform it into its idx representation
            sequence = text[i:i + sequence_length]
            sequence = [word_to_index[char] for char in sequence]

            # Get word target
            # Then, transform it into its idx representation
            # target = text[i + sequence_length]
            # target = word_to_index[target]
            target = text[i+1: i + 1 + sequence_length] # longer than +1 token out
            target = [word_to_index[char] for char in target]

            # Save sequences and targets
            x.append(sequence)
            y.append(target)

            # except:
            #     sequence = [word_to_index['PAD']]*sequence_length
            #     x.append(sequence)
            #     y.append(sequence)

        x = np.array(x)
        y = np.array(y)

        return x, y

    def get_data(self, dataframe):
        X = list()
        Y = list()

        for i in range(len(dataframe)):
            input = '<NEWSONG> ' + dataframe.iloc[i]['lyrics']
            tokenized_input = self.tokenizer(input)
            x, y = self.build_sequences(text=tokenized_input, word_to_index=self.word_to_index,
                                        sequence_length=self.sequence_length, max_len=self.max_len)
            X.append(x)
            Y.append(y)

        inputs = [sequence for song in X for sequence in song]
        targets = [targ for song in Y for targ in song]
        return np.array(inputs), np.array(targets)

    def to_categorical(self, y, num_classes):
        """ 1-hot encodes a tensor from https://discuss.pytorch.org/t/is-there-something-like-keras-utils-to-categorical-in-pytorch/5960"""
        return np.eye(num_classes, dtype='uint8')[y]

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        x = self.inputs[index]
        y = self.targets[index]
        y = self.to_categorical(y=y, num_classes=len(self.uniq_words)) # equiv to n_vocab
        return torch.tensor(x), torch.tensor(y)

class Model(nn.Module):
    def __init__(self, uniq_words, max_len, embedding_dim, embedding_matrix):
        super(Model, self).__init__()
        self.hidden_dim = 128
        self.embedding_dim = embedding_dim
        self.num_layers = 3
        # n_vocab = len(dataset.uniq_words)
        # self.max_len = dataset.max_len
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
        #self.softmax = nn.Softmax(dim=1)
        self.softmax = nn.Softmax2d()

    def forward(self, x, hidden):
        # batch_size = x.size(0)
        embed = self.embedding(x)
        output, hidden = self.lstm(embed, hidden)
        out = self.fc(output)
        # out = out[:, -1, :] # keeps only last subtensor tensor; likely want to use attention mechanism to create linear combo of all
        # out = self.softmax(out)
        return out, hidden

    def init_hidden(self, batch_size):
        h0 = torch.zeros((self.num_layers, batch_size, self.hidden_dim)).to(device)
        c0 = torch.zeros((self.num_layers, batch_size, self.hidden_dim)).to(device)
        return h0, c0

#--------------MODEL FUNCTIONS----------------
def train(train_dataset, val_dataset, model, batch_size, max_epochs, seq_len):
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, drop_last=True)
    criterion = nn.CrossEntropyLoss()
    #criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    all_loss = []
    for epoch in range(max_epochs):
        train_losses = []
        val_losses = []
        train_batch_count = 0
        val_batch_count = 0
        model.train()
        state_h, state_c = model.init_hidden(batch_size)
        for batch in train_dataloader:
            optimizer.zero_grad()
            X = batch[0].to(device)
            Y = batch[1].to(device)
            y_pred, (state_h, state_c) = model(X, (state_h, state_c))
            # loss = criterion(y_pred.transpose(1, 2), Y)
            loss = seq_len * criterion(y_pred, Y.float())
            state_h = state_h.detach()
            state_c = state_c.detach()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            print({ 'epoch': epoch, 'batch': train_batch_count, 'train loss': loss.item() })
            train_batch_count += 1
        model.eval()
        val_state_h, val_state_c = model.init_hidden(batch_size)
        for batch in val_dataloader:
            X = batch[0].to(device)
            Y = batch[1].to(device)
            y_pred, (val_state_h, val_state_c) = model(X, (val_state_h, val_state_c))
            # loss = criterion(y_pred.transpose(1, 2), Y)
            val_loss = criterion(y_pred, Y.float())
            val_state_h = val_state_h.detach()
            val_state_c = val_state_c.detach()
            val_losses.append(val_loss.item())
            print({'epoch': epoch, 'batch': val_batch_count, 'val loss': val_loss.item()})
            val_batch_count += 1


        epoch_train_loss = np.mean(train_losses)
        epoch_val_loss = np.mean(val_losses)
        all_loss.append(epoch_val_loss)
        print(f'Epoch {epoch}')
        print(f'train_loss : {epoch_train_loss} val_loss : {epoch_val_loss}')
        best_loss = max(all_loss)
        if epoch_val_loss <= best_loss:
            torch.save(model.state_dict(), "model_1.pt")
            print('model saved!')

#---------LOAD DATA--------------------
df = pd.read_csv('df_LSTM.csv', index_col=0)
df_copy = df.copy()
df_copy.reset_index(drop=True, inplace=True)
df_copy = df.iloc[0:500]

# create word dictionary for all datasets
words = load_words(df_copy)
uniq_words = get_uniq_words(words)
word_to_index = {word: index for index, word in enumerate(uniq_words)}
embedding_matrix = embedding_for_vocab(filepath=filepath, word_index=word_to_index, embedding_dim=embedding_dim)

# split data
train_, test_ = train_test_split(df_copy, train_size=0.8, random_state=SEED)
train_, val_ = train_test_split(train_, train_size=0.8, random_state=SEED)

# export datasets
train_.to_csv('train_data_m1.csv')
val_.to_csv('val_data_m1.csv')
test_.to_csv('test_data_m1.csv')

#-------------MODEL PREP----------------
train_dataset = Dataset(dataframe=train_, sequence_length=SEQUENCE_LEN, tokenizer=tokenizer, batch_size=BATCH_SIZE, max_len=MAX_LEN, words=words, uniq_words=uniq_words)
val_dataset = Dataset(dataframe=val_, sequence_length=SEQUENCE_LEN, tokenizer=tokenizer, batch_size=BATCH_SIZE, max_len=MAX_LEN, words=words, uniq_words=uniq_words)
test_dataset = Dataset(dataframe=test_, sequence_length=SEQUENCE_LEN, tokenizer=tokenizer, batch_size=BATCH_SIZE, max_len=MAX_LEN, words=words, uniq_words=uniq_words)
model = Model(uniq_words=uniq_words, max_len=MAX_LEN, embedding_dim=embedding_dim, embedding_matrix=embedding_matrix).to(device)

#------------MODEL TRAIN----------------
train(train_dataset=train_dataset, val_dataset=val_dataset, model=model, batch_size=BATCH_SIZE, max_epochs=EPOCHS, seq_len=SEQUENCE_LEN)
