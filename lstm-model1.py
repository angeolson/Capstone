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
EPOCHS = 50
MAX_LEN = 250
SEQUENCE_LEN = 4
LR = 0.001
TRUNCATE = False
DF_TRUNCATE_LB = 0  # lower bound to truncate data
DF_TRUNCATE_UB = 1000  # upper bound to truncate data
Iterative_Train = False  # False if training model from scratch, True if fine-tuning
single_token_output = False  # True if only want to look at last word logits
load_model = f'model-1-{DF_TRUNCATE_LB}.pt'
save_model = 'model-1-hs256.pt'
filepath_for_losses = 'epoch_losses_m1_hs256.csv'

embedding_dim = 200  # set = 50 for the 50d file, eg.
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
    unique_words_padding = ['<PAD>', '[BOS]', '[EOS]'] + unique_words
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
        max_len,
        words,
        uniq_words,
        single_token_output
    ):
        self.dataframe = dataframe
        self.single_token_output = single_token_output
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
        self.max_len = max_len
        self.inputs, self.targets = self.get_data(self.dataframe)
        self.batch_size = len(self.inputs) // len(self.dataframe)

    def __len__(self):
        return len(self.inputs)

    def build_sequences(self, text, word_to_index, sequence_length, max_len, single_token_output=True):
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
                text.append('<PAD>')
        for i in range(max_len - (sequence_length + 1)):
            # try:
            # Get window of chars from text
            # Then, transform it into its idx representation
            sequence = text[i:i + sequence_length]
            sequence = [word_to_index[char] for char in sequence]

            # Get word target
            # Then, transform it into its idx representation

            if single_token_output is True:
                target = text[i + 1 + sequence_length]
                target = word_to_index[target]
            else:
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
            # input = '<NEWSONG> ' + dataframe.iloc[i]['lyrics']
            input = '[BOS] ' + dataframe.iloc[i]['lyrics'] + ' [EOS]'
            #input = dataframe.iloc[i]['lyrics']
            tokenized_input = self.tokenizer(input)
            x, y = self.build_sequences(text=tokenized_input, word_to_index=self.word_to_index,
                                        sequence_length=self.sequence_length, max_len=self.max_len, single_token_output=self.single_token_output)
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
        #y = self.to_categorical(y=y, num_classes=len(self.uniq_words)) # equiv to n_vocab
        return torch.tensor(x), torch.tensor(y)

class Model(nn.Module):
    def __init__(self, uniq_words, max_len, embedding_dim, embedding_matrix, single_token_output=True):
        super(Model, self).__init__()
        self.hidden_dim = 256
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

#--------------MODEL FUNCTIONS----------------
#def train(train_dataset, val_dataset, model, batch_size, max_epochs, seq_len):
def train(train_dataset, val_dataset, model, max_epochs, seq_len):
    all_val_loss = []
    all_train_loss = []
    train_batch_size = train_dataset.batch_size
    val_batch_size = val_dataset.batch_size
    train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=train_batch_size, drop_last=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(max_epochs):
        train_losses = []
        val_losses = []
        train_batch_count = 0
        val_batch_count = 0
        model.train()
        state_h, state_c = model.init_hidden(train_batch_size)
        for batch in train_dataloader:
            optimizer.zero_grad()
            X = batch[0].to(device)
            Y = batch[1].to(device)
            y_pred, (state_h, state_c) = model(X, (state_h, state_c))
            outputs = y_pred.view(seq_len * train_batch_size, len(train_dataset.uniq_words))
            targets = Y.view(-1)
            if len(targets) == seq_len * train_batch_size:
                loss = criterion(outputs, targets)
                state_h = state_h.detach()
                state_c = state_c.detach()
                loss.backward()
                train_losses.append(loss.item())
                print({'epoch': epoch, 'batch': train_batch_count, 'train loss': loss.item()})
                train_batch_count += 1
            optimizer.step()
        model.eval()
        val_state_h, val_state_c = model.init_hidden(val_batch_size)
        for batch in val_dataloader:
            X = batch[0].to(device)
            Y = batch[1].to(device)
            y_pred, (val_state_h, val_state_c) = model(X, (val_state_h, val_state_c))
            outputs = y_pred.view(seq_len * val_batch_size, len(val_dataset.uniq_words))
            targets = Y.view(-1)
            if len(targets) == seq_len * val_batch_size:
                val_loss = criterion(outputs, targets)
                val_state_h = val_state_h.detach()
                val_state_c = val_state_c.detach()
                val_losses.append(val_loss.item())
                print({'epoch': epoch, 'batch': val_batch_count, 'val loss': val_loss.item()})
                val_batch_count += 1

        epoch_train_loss = np.mean(train_losses)
        epoch_val_loss = np.mean(val_losses)
        all_val_loss.append(epoch_val_loss)
        all_train_loss.append(epoch_train_loss)
        print(f'Epoch {epoch}')
        print(f'train_loss : {epoch_train_loss} val_loss : {epoch_val_loss}')
        best_loss = min(all_val_loss)
        if epoch_val_loss <= best_loss:
            torch.save(model.state_dict(), save_model)
            print('model saved!')
        losses_df = pd.DataFrame()
        losses_df['val_loss'] = all_val_loss
        losses_df['train_loss'] = all_train_loss
        losses_df.to_csv(filepath_for_losses)

#---------LOAD DATA--------------------
df = pd.read_csv('df_LSTM.csv', index_col=0)
df_copy = df.copy()
df_copy.reset_index(drop=True, inplace=True)

# create word dictionary for all datasets
words = load_words(df_copy)
uniq_words = get_uniq_words(words)
word_to_index = {word: index for index, word in enumerate(uniq_words)}
embedding_matrix = embedding_for_vocab(filepath=filepath, word_index=word_to_index, embedding_dim=embedding_dim)

# read data to use
df_train = pd.read_csv('training.csv', index_col=0)
df_train.reset_index(drop=True, inplace=True)

df_val = pd.read_csv('validation.csv', index_col=0)
df_val.reset_index(drop=True, inplace=True)

# truncate; 0.2*250 = 50
if TRUNCATE is True:
    train_ = df_train.iloc[DF_TRUNCATE_LB:DF_TRUNCATE_UB]
    val_ = df_val.iloc[DF_TRUNCATE_LB:int(DF_TRUNCATE_UB*0.2)]
else:
    train_ = df_train.copy()
    val_ = df_val.copy()

#-------------MODEL PREP----------------
train_dataset = Dataset(dataframe=train_, sequence_length=SEQUENCE_LEN, tokenizer=tokenizer, max_len=MAX_LEN, words=words, uniq_words=uniq_words, single_token_output=single_token_output)
val_dataset = Dataset(dataframe=val_, sequence_length=SEQUENCE_LEN, tokenizer=tokenizer, max_len=MAX_LEN, words=words, uniq_words=uniq_words, single_token_output=single_token_output)

model = Model(uniq_words=uniq_words, max_len=MAX_LEN, embedding_dim=embedding_dim, embedding_matrix=embedding_matrix, single_token_output=single_token_output).to(device)
if Iterative_Train is True:
    model.load_state_dict(torch.load(load_model, map_location=device))

#------------MODEL TRAIN----------------
# train(train_dataset=train_dataset, val_dataset=val_dataset, model=model, batch_size=BATCH_SIZE, max_epochs=EPOCHS, seq_len=SEQUENCE_LEN)
train(train_dataset=train_dataset, val_dataset=val_dataset, model=model, max_epochs=EPOCHS, seq_len=SEQUENCE_LEN)